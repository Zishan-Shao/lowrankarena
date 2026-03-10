#!/usr/bin/env python3
"""Decode speed benchmark: ASVD baseline vs ASVD + FlashSVD kernels.

Mirrors the flow of baselines/SVD-LLM/bench_flashsvd_vs_svd_decode.py.

Two modes per run:
  [ASVD baseline]  — SVDLinear forward, standard StaticCache
  [ASVD+FlashSVD]  — same model, flashsvd_wrapper applied, standard StaticCache

Usage (compress fresh):
  python bench_asvd_flashsvd.py --model_id jeffwan/llama-7b-hf \
      --param_ratio_target 0.5 --dtype bf16 --prompt_len 512 --new_tokens 128

Usage (load pre-saved checkpoint):
  python bench_asvd_flashsvd.py --checkpoint ./checkpoints/asvd_llama7b_0.5.pt \
      --dtype bf16 --prompt_len 512 --new_tokens 128

Save checkpoint after compression:
  python bench_asvd_flashsvd.py --model_id jeffwan/llama-7b-hf \
      --param_ratio_target 0.5 --save_path ./checkpoints --dtype bf16 ...
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

# Register repo root so src.kernels.decoder.* is importable
_HERE = os.path.dirname(os.path.abspath(__file__))          # baselines/ASVD/
_REPO = os.path.dirname(os.path.dirname(_HERE))             # lowrankarena/
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
# Also add ASVD dir so local imports (datautils, etc.) work
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# SVD-LLM evaluater for decode timing (reused)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "SVD-LLM"))
from evaluater import decode_kvcache_eval

from flashsvd_wrapper import apply_flashsvd_to_asvd_model


# ── helpers ───────────────────────────────────────────────────────────────────

def _dtype_from_name(name: str) -> torch.dtype | None:
    raw = str(name).strip().lower()
    if raw == "auto":
        return None
    if raw in {"fp16", "float16", "half"}:
        return torch.float16
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _cast_model(model, dtype_name: str):
    raw = str(dtype_name).strip().lower()
    if raw == "fp32":
        return model.float()
    if raw == "fp16":
        return model.half()
    if raw == "bf16":
        return model.to(dtype=torch.bfloat16)
    if raw == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return model.to(dtype=torch.bfloat16)
        return model.half()
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _compress_model(args):
    """Run ASVD compression and return (model, tokenizer)."""
    from datautils import get_calib_data
    from act_aware_utils import calib_input_distribution, calib_fisher_info
    from sensitivity import calib_sensitivity_ppl, calib_sensitivity_stable_rank
    from binary_search import binary_search_truncation_rank

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"[ASVD] Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    print(f"[ASVD] Calibrating (dataset={args.calib_dataset}, n={args.n_calib_samples}) ...")
    calib_loader = get_calib_data(
        args.calib_dataset, tokenizer, args.model_id, args.n_calib_samples,
        seed=args.seed, use_bos=args.use_bos,
    )
    if "fisher" in args.scaling_method:
        calib_fisher_info(model, calib_loader, args.use_cache)
    if "abs" in args.scaling_method:
        calib_input_distribution(model, calib_loader, args.scaling_method, args.use_cache)

    if args.sensitivity_metric == "ppl":
        sensitivity = calib_sensitivity_ppl(model, calib_loader, args, args.use_cache)
    else:
        sensitivity = calib_sensitivity_stable_rank(model, calib_loader, args, args.use_cache)

    print(f"[ASVD] Searching truncation ranks (param_ratio={args.param_ratio_target}) ...")
    binary_search_truncation_rank(model, sensitivity, calib_loader, args)

    return model, tokenizer


def _load_checkpoint(path: str):
    print(f"[ASVD] Loading checkpoint from {path} ...")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["model"], obj["tokenizer"]


def _force_uniform_qkv(model, target_rank: int = 0) -> int:
    """Make q/k/v projections all SVDLinear with equal rank.

    - Plain nn.Linear projections are replaced with SVDLinear at target_rank.
    - SVDLinear projections whose rank != target_rank are re-compressed.
    - If target_rank=0, uses the minimum rank found among existing SVDLinear
      projections in the model.

    Returns the number of projections replaced/re-compressed.
    """
    from modules.svd_linear import SVDLinear

    def _is_svd(m):
        return isinstance(m, SVDLinear)

    def _rank(m: SVDLinear) -> int:
        return int(m.BLinear.weight.shape[0])

    def _to_linear(m: SVDLinear) -> torch.nn.Linear:
        """Reconstruct a plain nn.Linear from an SVDLinear (approximate)."""
        W = m.ALinear.weight @ m.BLinear.weight  # [out, in]
        lin = torch.nn.Linear(m.BLinear.in_features, m.ALinear.out_features,
                              bias=m.ALinear.bias is not None)
        lin.weight = torch.nn.Parameter(W)
        if m.ALinear.bias is not None:
            lin.bias = torch.nn.Parameter(m.ALinear.bias.clone())
        return lin

    try:
        layers = model.model.layers
    except AttributeError:
        return 0

    # Auto-detect target rank from minimum existing SVDLinear rank
    if target_rank == 0:
        for layer in layers:
            attn = getattr(layer, 'self_attn', None)
            if attn is None:
                continue
            for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
                if _is_svd(proj):
                    target_rank = min(target_rank or _rank(proj), _rank(proj))
        if target_rank == 0:
            print("[force_uniform_qkv] no SVDLinear found, skipping")
            return 0

    print(f"[force_uniform_qkv] target_rank={target_rank}")
    patched = 0
    for layer in layers:
        attn = getattr(layer, 'self_attn', None)
        if attn is None:
            continue
        for name in ['q_proj', 'k_proj', 'v_proj']:
            proj = getattr(attn, name)
            # Get base nn.Linear (reconstruct if already SVDLinear)
            if _is_svd(proj):
                if _rank(proj) == target_rank:
                    continue  # already correct rank
                proj = _to_linear(proj)  # reconstruct for re-compression
            n_params = proj.weight.numel()
            in_out = proj.in_features + proj.out_features
            param_ratio = target_rank * in_out / n_params
            svd = SVDLinear.from_linear(proj, param_ratio=param_ratio, act_aware=False)
            setattr(attn, name, svd)
            patched += 1

    print(f"[force_uniform_qkv] replaced/recompressed {patched} projections")
    return patched


def _bench_one_mode(
    checkpoint_path: str | None,
    *,
    mode: str,
    args,
) -> dict:
    """Load model fresh, run one benchmark mode, then release GPU memory.

    Mirrors SVD-LLM's _bench_one_mode: each call gets a clean GPU slate.
    mode: "baseline" | "flashsvd"
    """
    assert mode in ("baseline", "flashsvd")

    # Fresh load from checkpoint
    model, tokenizer = _load_checkpoint(checkpoint_path)
    model.eval()
    model = _cast_model(model, args.dtype)
    model = model.to(args.device)

    if mode == "flashsvd":
        if args.force_uniform_qkv:
            _force_uniform_qkv(model, target_rank=args.target_rank)
        apply_flashsvd_to_asvd_model(model)

    try:
        result = decode_kvcache_eval(
            model,
            prompt_len=args.prompt_len,
            new_tokens=args.new_tokens,
            warmup=args.warmup,
            batch_size=args.batch_size,
            device=args.device,
            lowrank_cache=False,
            flashsvd_dense_cache=(mode == "flashsvd"),
            baseline_dense_kvcache=False,
            profile_decode=False,
        )
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available() and "cuda" in str(args.device):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser("ASVD baseline vs ASVD+FlashSVD decode benchmark")

    # Model source (one of these two groups must be provided)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--checkpoint", type=str, default=None,
                     help="Path to pre-saved ASVD checkpoint (.pt)")
    src.add_argument("--model_id", type=str, default=None,
                     help="HuggingFace model ID to compress fresh")

    # ASVD compression args (only used with --model_id)
    ap.add_argument("--param_ratio_target", type=float, default=0.5)
    ap.add_argument("--act_aware", action="store_true", default=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n_calib_samples", type=int, default=32)
    ap.add_argument("--calib_dataset", type=str, default="wikitext2",
                    choices=["wikitext2", "c4", "ptb", "alpaca", "selfgen"])
    ap.add_argument("--scaling_method", type=str, default="abs_mean",
                    choices=["abs_mean", "abs_max", "fisher", "fisher_abs_mean"])
    ap.add_argument("--sensitivity_metric", type=str, default="ppl",
                    choices=["ppl", "stable_rank"])
    ap.add_argument("--use_cache", action="store_true", default=False)
    ap.add_argument("--use_bos", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight_quant", type=str, default="none")
    ap.add_argument("--ppl_target", type=float, default=-1)
    ap.add_argument("--compress_kv_cache", action="store_true")
    ap.add_argument("--kv_cache_ratio_target", type=float, default=-1)
    ap.add_argument("--sigma_fuse", type=str, default="UV", choices=["U", "V", "UV"])
    ap.add_argument("--rank_align", type=int, default=1)

    # Save compressed model
    ap.add_argument("--save_path", type=str, default=None,
                    help="Directory to save compressed checkpoint")

    # Benchmark args
    ap.add_argument("--dtype", type=str, default="auto",
                    choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--prompt_len", type=int, default=512)
    ap.add_argument("--new_tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--skip_baseline", action="store_true")
    ap.add_argument("--skip_flashsvd", action="store_true")
    ap.add_argument("--force_uniform_qkv", action="store_true",
                    help="Before FlashSVD, make q/k/v all SVDLinear with equal rank")
    ap.add_argument("--target_rank", type=int, default=0,
                    help="Target rank for --force_uniform_qkv (0=auto: min existing rank)")
    ap.add_argument("--sdp_backend", type=str, default=None,
                    choices=["flash", "mem_efficient", "math"],
                    help="Lock PyTorch SDPA backend for both baseline and FlashSVD runs")

    args = ap.parse_args()

    if args.checkpoint is None and args.model_id is None:
        ap.error("Provide --checkpoint or --model_id")

    if args.sdp_backend is not None:
        torch.backends.cuda.enable_flash_sdp(args.sdp_backend == "flash")
        torch.backends.cuda.enable_mem_efficient_sdp(args.sdp_backend == "mem_efficient")
        torch.backends.cuda.enable_math_sdp(args.sdp_backend == "math")
        print(f"[sdp] locked backend: {args.sdp_backend}")

    print("==== ASVD vs ASVD+FlashSVD Decode Benchmark ====")
    print(
        f"Config: prompt_len={args.prompt_len} new_tokens={args.new_tokens} "
        f"warmup={args.warmup} batch={args.batch_size} dtype={args.dtype} device={args.device}"
    )

    # ── Compress (if needed) and save to a temp file for reload isolation ──────
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        model, tokenizer = _compress_model(args)
        Path(args.save_path or ".").mkdir(parents=True, exist_ok=True)
        model_name = args.model_id.replace("/", "_").replace("-", "_")
        save_dir = Path(args.save_path) if args.save_path else Path(".")
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(save_dir / f"{model_name}_asvd_ratio{args.param_ratio_target}.pt")
        torch.save({"model": model, "tokenizer": tokenizer}, checkpoint_path)
        print(f"[ASVD] Checkpoint saved to {checkpoint_path}")
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available() and "cuda" in str(args.device):
            torch.cuda.empty_cache()

    results: dict[str, dict] = {}

    # ── Baseline: fresh model load, run, release ───────────────────────────────
    if not args.skip_baseline:
        print("\n[ASVD baseline]")
        results["baseline"] = _bench_one_mode(checkpoint_path, mode="baseline", args=args)

    # ── FlashSVD: fresh model load, patch, run, release ───────────────────────
    if not args.skip_flashsvd:
        print("\n[ASVD+FlashSVD]")
        results["flashsvd"] = _bench_one_mode(checkpoint_path, mode="flashsvd", args=args)

    # ── Summary ───────────────────────────────────────────────────────────────
    if "baseline" in results and "flashsvd" in results:
        base = results["baseline"]
        flash = results["flashsvd"]
        speedup = float(base["decode_ms_per_token"]) / max(float(flash["decode_ms_per_token"]), 1e-9)
        print("\n---- Summary ----")
        print(f"ASVD baseline : {float(base['decode_ms_per_token']):.3f} ms/token | {float(base['decode_tok_s']):,.0f} tok/s")
        print(f"ASVD+FlashSVD : {float(flash['decode_ms_per_token']):.3f} ms/token | {float(flash['decode_tok_s']):,.0f} tok/s")
        print(f"FlashSVD speedup vs ASVD: {speedup:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
