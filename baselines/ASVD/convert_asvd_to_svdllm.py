#!/usr/bin/env python3
"""Convert an ASVD-compressed LlamaForCausalLM checkpoint to SVD-LLM format.

ASVD replaces each nn.Linear with SVDLinear(BLinear, ALinear):
  forward: ALinear(BLinear(x))  =>  x @ B^T @ A^T
  BLinear.weight  [rank, in ]  →  v_proj.weight
  ALinear.weight  [out, rank]  →  u_proj.weight

SVD-LLM uses named splits:
  q_v_proj / q_u_proj,  k_v_proj / k_u_proj,  v_v_proj / v_u_proj,  o_v_proj / o_u_proj
  gate_v_proj / gate_u_proj,  up_v_proj / up_u_proj,  down_v_proj / down_u_proj

The weight layout is identical; only names differ — so conversion is direct assignment.

Rank-uniformity constraint for SVD-LLM decode kernels:
  - Attention : R_q == R_k == R_v  (decode kernel; R_o independent)
  - MLP       : R_gate == R_up     (dual-split kernel; R_down independent)
  If ranks differ, the minimum is taken and top-rank truncation is applied.

Usage
-----
  python convert_asvd_to_svdllm.py --checkpoint <asvd.pt> --out <out.pt>

  # then benchmark with existing SVD-LLM bench script:
  cd ../SVD-LLM
  python bench_flashsvd_vs_svd_decode.py --model_path <out.pt> --mode flashsvd ...
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_SVDLLM = os.path.join(os.path.dirname(_HERE), "SVD-LLM")

for p in [_REPO, _SVDLLM]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_svd(m: nn.Module) -> bool:
    return type(m).__name__ in ("SVDLinear", "SVDTransformLayer")


def _rank_of(m: nn.Module) -> int:
    """Return the compression rank, or full-rank for nn.Linear."""
    t = type(m).__name__
    if t == "SVDLinear":
        # ASVD: BLinear.weight [rank, in]
        return int(m.BLinear.weight.shape[0])
    if t == "SVDTransformLayer":
        # DobiSVD: ALinear.weight [rank, in]
        return int(m.ALinear.weight.shape[0])
    # plain nn.Linear — not compressed
    return min(m.in_features, m.out_features)


def _new_linear(weight: torch.Tensor, bias: torch.Tensor | None = None) -> nn.Linear:
    """Create an nn.Linear whose weight is exactly `weight` (no copy if not needed)."""
    out_f, in_f = weight.shape
    lin = nn.Linear(in_f, out_f, bias=bias is not None)
    lin.weight = nn.Parameter(weight.contiguous())
    if bias is not None:
        lin.bias = nn.Parameter(bias.contiguous())
    return lin


def _truncate(weight: torch.Tensor, new_rank: int, dim: int) -> torch.Tensor:
    """Keep only the first `new_rank` slices along `dim`."""
    slices = [slice(None)] * weight.ndim
    slices[dim] = slice(0, new_rank)
    return weight[tuple(slices)].contiguous()


def _get_uv(svd_linear: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (u_weight [out, R], v_weight [R, in]) from an SVDLinear or SVDTransformLayer.

    ASVD SVDLinear layout:
        ALinear.weight [out, R]  → U
        BLinear.weight [R,   in] → V

    DobiSVD SVDTransformLayer layout (names are swapped relative to ASVD):
        ALinear.weight [R,   in] → V
        BLinear.weight [out, R]  → U
    """
    if type(svd_linear).__name__ == "SVDTransformLayer":
        return svd_linear.BLinear.weight.data, svd_linear.ALinear.weight.data
    return svd_linear.ALinear.weight.data, svd_linear.BLinear.weight.data


def _get_uv_at_rank(
    svd_linear: nn.Module, target_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return U/V truncated to target_rank (noop if already at that rank)."""
    u, v = _get_uv(svd_linear)
    r = u.shape[1]  # current rank  (u: [out, R], v: [R, in])
    if r == target_rank:
        return u, v
    # truncate rank dimension
    return _truncate(u, target_rank, dim=1), _truncate(v, target_rank, dim=0)


# ---------------------------------------------------------------------------
# Per-layer conversion
# ---------------------------------------------------------------------------

def _convert_attention(
    hf_attn: nn.Module,
    config,
    layer_idx: int,
) -> nn.Module:
    """Convert HF LlamaAttention (with SVDLinear projections) → SVD_LlamaAttention."""
    from flashsvd_component.svd_llama import SVD_LlamaAttention

    q, k, v, o = hf_attn.q_proj, hf_attn.k_proj, hf_attn.v_proj, hf_attn.o_proj

    if not (_is_svd(q) and _is_svd(k) and _is_svd(v)):
        print(f"  [layer {layer_idx}] attention: some projections not SVDLinear — skipping")
        return hf_attn

    # Uniform rank for q/k/v (decode kernel requirement)
    rank_q, rank_k, rank_v = _rank_of(q), _rank_of(k), _rank_of(v)
    rank_attn = min(rank_q, rank_k, rank_v)
    if rank_q != rank_k or rank_q != rank_v:
        print(
            f"  [layer {layer_idx}] attention ranks q={rank_q} k={rank_k} v={rank_v} → "
            f"truncating to {rank_attn}"
        )

    q_u, q_v = _get_uv_at_rank(q, rank_attn)
    k_u, k_v = _get_uv_at_rank(k, rank_attn)
    v_u, v_v = _get_uv_at_rank(v, rank_attn)

    # o_proj rank is independent
    if _is_svd(o):
        o_u, o_v = _get_uv(o)
    else:
        # plain nn.Linear — do thin SVD to get factorized form
        print(f"  [layer {layer_idx}] o_proj is plain Linear — computing thin SVD")
        W = o.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r_o = min(W.shape[0], W.shape[1])
        sqS = S[:r_o].sqrt()
        o_u = (U[:, :r_o] * sqS).to(q_u.dtype)
        o_v = (Vh[:r_o] * sqS.unsqueeze(1)).to(q_u.dtype)

    # Instantiate SVD_LlamaAttention (ratio=0.5 is a placeholder; we override weights)
    cfg = config
    cfg_layer_idx_orig = getattr(cfg, "layer_idx", None)
    cfg.layer_idx = layer_idx
    svd_attn = SVD_LlamaAttention(config=cfg, ratio=0.5)
    if cfg_layer_idx_orig is None:
        try:
            delattr(cfg, "layer_idx")
        except AttributeError:
            pass
    else:
        cfg.layer_idx = cfg_layer_idx_orig

    # Replace sub-modules with correctly-sized Linears
    svd_attn.q_u_proj = _new_linear(q_u)
    svd_attn.q_v_proj = _new_linear(q_v)
    svd_attn.k_u_proj = _new_linear(k_u)
    svd_attn.k_v_proj = _new_linear(k_v)
    svd_attn.v_u_proj = _new_linear(v_u)
    svd_attn.v_v_proj = _new_linear(v_v)
    svd_attn.o_u_proj = _new_linear(o_u)
    svd_attn.o_v_proj = _new_linear(o_v)

    # Update rank bookkeeping used by the decode path
    svd_attn.low_rank = rank_attn
    svd_attn.layer_idx = layer_idx

    # Carry over RoPE from original layer if present
    if hasattr(hf_attn, "rotary_emb"):
        svd_attn.rotary_emb = hf_attn.rotary_emb
        svd_attn.flash_attn.rotary_emb = hf_attn.rotary_emb

    return svd_attn


def _convert_mlp(
    hf_mlp: nn.Module,
    config,
    layer_idx: int,
) -> nn.Module:
    """Convert HF LlamaMLP (with SVDLinear projections) → SVD_LlamaMLP."""
    from flashsvd_component.svd_llama import SVD_LlamaMLP

    gate, up, down = hf_mlp.gate_proj, hf_mlp.up_proj, hf_mlp.down_proj

    if not (_is_svd(gate) and _is_svd(up) and _is_svd(down)):
        print(f"  [layer {layer_idx}] mlp: some projections not SVDLinear — skipping")
        return hf_mlp

    # Uniform rank for gate/up (dual-split decode kernel requirement)
    rank_gate, rank_up = _rank_of(gate), _rank_of(up)
    rank_mlp = min(rank_gate, rank_up)
    if rank_gate != rank_up:
        print(
            f"  [layer {layer_idx}] mlp ranks gate={rank_gate} up={rank_up} → "
            f"truncating to {rank_mlp}"
        )

    gate_u, gate_v = _get_uv_at_rank(gate, rank_mlp)
    up_u, up_v = _get_uv_at_rank(up, rank_mlp)
    down_u, down_v = _get_uv(down)

    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    hidden_act = str(getattr(config, "hidden_act", "silu"))

    svd_mlp = SVD_LlamaMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        ratio=0.5,  # placeholder
    )

    svd_mlp.gate_u_proj = _new_linear(gate_u)
    svd_mlp.gate_v_proj = _new_linear(gate_v)
    svd_mlp.up_u_proj = _new_linear(up_u)
    svd_mlp.up_v_proj = _new_linear(up_v)
    svd_mlp.down_u_proj = _new_linear(down_u)
    svd_mlp.down_v_proj = _new_linear(down_v)

    return svd_mlp


# ---------------------------------------------------------------------------
# Full model conversion
# ---------------------------------------------------------------------------

def convert_asvd_to_svdllm(model: nn.Module) -> nn.Module:
    """In-place convert all LlamaDecoderLayer children from ASVD → SVD-LLM format."""
    try:
        layers = model.model.layers
    except AttributeError:
        raise ValueError("model.model.layers not found — expected LlamaForCausalLM")

    config = model.config
    n_converted_attn = 0
    n_converted_mlp = 0

    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)

        if attn is not None:
            new_attn = _convert_attention(attn, config, layer_idx=i)
            if new_attn is not attn:
                layer.self_attn = new_attn
                n_converted_attn += 1

        if mlp is not None:
            new_mlp = _convert_mlp(mlp, config, layer_idx=i)
            if new_mlp is not mlp:
                layer.mlp = new_mlp
                n_converted_mlp += 1

    print(
        f"[convert] done — {n_converted_attn} attention layers, "
        f"{n_converted_mlp} MLP layers converted to SVD-LLM format"
    )
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser("Convert ASVD checkpoint → SVD-LLM format")
    ap.add_argument("--checkpoint", required=True, help="Path to ASVD .pt checkpoint")
    ap.add_argument("--out", required=True, help="Output .pt path for SVD-LLM checkpoint")
    ap.add_argument(
        "--bench", action="store_true",
        help="Run a quick decode benchmark after conversion (requires CUDA)"
    )
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--prompt_len", type=int, default=512)
    ap.add_argument("--new_tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()

    print(f"[convert] Loading ASVD checkpoint: {args.checkpoint}")
    obj = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = obj["model"]
    tokenizer = obj.get("tokenizer", None)

    model.eval()

    print("[convert] Converting ASVD → SVD-LLM format ...")
    convert_asvd_to_svdllm(model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_obj = {"model": model}
    if tokenizer is not None:
        save_obj["tokenizer"] = tokenizer
    torch.save(save_obj, str(out_path))
    print(f"[convert] Saved to {out_path}")

    if args.bench:
        if not torch.cuda.is_available():
            print("[bench] No CUDA — skipping benchmark")
            return 0

        sys.path.insert(0, _SVDLLM)
        from evaluater import decode_kvcache_eval

        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        model = model.to(dtype=dtype_map[args.dtype]).cuda()

        # SVD baseline (fallback, no FlashSVD kernels)
        print("\n[bench] SVD baseline (no FlashSVD kernels) ...")
        os.environ["SVDLLM_FLASH_FALLBACK"] = "1"
        os.environ["FLASH_SVD_DISABLE_FFN"] = "1"
        os.environ["FLASH_SVD_BASELINE_DENSE_KVCACHE"] = "1"
        os.environ["FLASH_SVD_REFERENCE_DENSE_ATTN"] = "1"
        r_base = decode_kvcache_eval(
            model,
            prompt_len=args.prompt_len, new_tokens=args.new_tokens,
            warmup=args.warmup, batch_size=args.batch_size,
            device="cuda", lowrank_cache=False,
            flashsvd_dense_cache=False, baseline_dense_kvcache=True,
        )
        print(f"  SVD baseline : {float(r_base['decode_ms_per_token']):.2f} ms/tok | "
              f"{float(r_base['decode_tok_s']):.0f} tok/s")

        # FlashSVD (dense KV cache + FA2)
        print("\n[bench] FlashSVD (dense KV cache + FA2 decode) ...")
        os.environ.pop("SVDLLM_FLASH_FALLBACK", None)
        os.environ.pop("FLASH_SVD_DISABLE_FFN", None)
        os.environ["FLASH_SVD_ENABLE_DENSE_ATTN_DECODE"] = "1"
        os.environ["FLASH_SVD_BASELINE_DENSE_KVCACHE"] = "0"
        os.environ["FLASH_SVD_REFERENCE_DENSE_ATTN"] = "0"
        os.environ["FLASH_SVD_FFN_BACKEND"] = "dual_split_cublas"
        r_flash = decode_kvcache_eval(
            model,
            prompt_len=args.prompt_len, new_tokens=args.new_tokens,
            warmup=args.warmup, batch_size=args.batch_size,
            device="cuda", lowrank_cache=False,
            flashsvd_dense_cache=True, baseline_dense_kvcache=False,
        )
        print(f"  FlashSVD     : {float(r_flash['decode_ms_per_token']):.2f} ms/tok | "
              f"{float(r_flash['decode_tok_s']):.0f} tok/s")

        speedup = float(r_base["decode_ms_per_token"]) / max(float(r_flash["decode_ms_per_token"]), 1e-9)
        print(f"\n  Speedup: {speedup:.2f}x")

        gc.collect()
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
