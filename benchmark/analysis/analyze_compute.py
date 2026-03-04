#!/usr/bin/env python3
"""
FLOP / Memory-Traffic / GPU-Utilization breakdown for SVD-compressed BERT.

Outputs per-layer and aggregate:
  - Useful FLOPs   (based on actual rank R)
  - Padding FLOPs  (overhead due to next_pow2(R) alignment for Triton)
  - Weight traffic  (bytes to load weight matrices)
  - Activation traffic (bytes to read/write activations)
  - Total memory traffic
  - Arithmetic Intensity  (FLOPs/byte, roofline position)
  - Achieved TFLOPS  (from measured latency)
  - MFU              (achieved / GPU peak)

Usage
-----
# Load a saved checkpoint and benchmark on sst2:
python benchmark/analyze_compute.py \
    --model_dir compressed_models/bert/sst2/svd_r256_naive \
    --task sst2 --backend flashsvd15 --dtype bf16

# Or compress on-the-fly (slower):
python benchmark/analyze_compute.py \
    --method fwsvd --rank 128 --task sst2 --backend flashsvd15 --dtype bf16
"""

import argparse
import math
import os
import sys
import time
import statistics

import torch

# ── repo root on path ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))  # lowrankarena/..
_LOWRANK = os.path.abspath(os.path.join(_HERE, "..", ".."))      # lowrankarena/
for _p in (_REPO, _LOWRANK):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ══════════════════════════════════════════════════════════════════════════════
# GPU peak specs (Tensor Core, BF16 / FP16)
# ══════════════════════════════════════════════════════════════════════════════
_GPU_PEAK_TFLOPS = {
    "a100": 312.0, "a10g": 125.0, "a10": 125.0,
    "v100": 125.0,
    "3090": 142.0, "rtx 3090": 142.0,
    "4090": 330.0, "rtx 4090": 330.0,
    "h100": 989.0, "h800": 700.0,
    "l40s": 733.0, "l40": 362.0,
    "3080": 119.0, "3070": 81.0,
}
# HBM bandwidth (GB/s)
_GPU_BANDWIDTH_GBS = {
    "a100": 2000.0, "a10g": 600.0, "a10": 600.0,
    "v100": 900.0,
    "3090": 936.0, "rtx 3090": 936.0,
    "4090": 1008.0, "rtx 4090": 1008.0,
    "h100": 3350.0, "h800": 2400.0,
    "l40s": 864.0, "l40": 864.0,
    "3080": 760.0, "3070": 448.0,
}

# Default alignment warn threshold per dtype.
# Rationale:
#   bf16 — 7 mantissa bits (~0.8% relative precision); different op-ordering
#           across backends can produce absolute logit diffs up to ~0.05.
#   fp16 — 10 mantissa bits (~0.1% relative); tighter than bf16.
#   fp32 — 23 mantissa bits; backends should agree to within rounding noise.
_ALIGN_WARN_THRESH_BY_DTYPE = {
    "bf16": 0.05,
    "fp16": 0.02,
    "fp32": 1e-3,
}


def _gpu_specs():
    """Return (peak_tflops, bandwidth_gbs, gpu_name) for the current device."""
    if not torch.cuda.is_available():
        return None, None, "cpu"
    name = torch.cuda.get_device_name(0).lower()
    peak = next((v for k, v in _GPU_PEAK_TFLOPS.items() if k in name), None)
    bw   = next((v for k, v in _GPU_BANDWIDTH_GBS.items() if k in name), None)
    return peak, bw, torch.cuda.get_device_name(0)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _dtype_bytes(dtype) -> int:
    return {
        torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
    }.get(dtype, 4)


def _git_commit() -> str:
    """Return current HEAD short commit hash, or '' if not in a git repo."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def _get_encoder_layers(model):
    if hasattr(model, "bert"):
        return model.bert.encoder.layer
    if hasattr(model, "roberta"):
        return model.roberta.encoder.layer
    raise RuntimeError("Unsupported architecture (no bert/roberta attribute)")


def _make_synthetic_loader(seq_len: int, batch_size: int, num_batches: int,
                            vocab_size: int = 30522, seed: int = 42):
    """
    Synthetic loader: random token IDs with all-1 attention mask (zero sequence padding).

    Purpose: isolates backend kernel performance from dataset-specific padding overhead.
    Use for the "fully-utilized input" profile in expB alongside the real-data profile.

    The collator returns the same dict structure as real loaders
    (input_ids, attention_mask, token_type_ids) so the model forward path is identical.
    No extra CPU preprocessing is triggered — the tensors are pre-generated once.

    Args:
        seq_len:    Sequence length (every position is an effective token).
        batch_size: Batch size.
        num_batches: Number of batches to pre-generate (warmup + measure × repeat + headroom).
        vocab_size:  Tokenizer vocab size (default BERT 30522).
                     Uses range [100, vocab_size-100] to avoid special tokens (CLS/SEP/PAD).
        seed:        RNG seed for reproducibility (default 42).
                     Performance is insensitive to specific token values,
                     but fixing the seed aids exact reproducibility.
    """
    from torch.utils.data import DataLoader, TensorDataset
    N = num_batches * batch_size
    rng = torch.Generator()
    rng.manual_seed(seed)
    # Random token IDs in [100, vocab_size-100]; avoids special tokens at vocab boundaries
    input_ids      = torch.randint(100, vocab_size - 100, (N, seq_len),
                                   dtype=torch.long, generator=rng)
    attention_mask = torch.ones(N, seq_len, dtype=torch.long)   # all tokens effective (0% padding)
    token_type_ids = torch.zeros(N, seq_len, dtype=torch.long)

    ds = TensorDataset(input_ids, attention_mask, token_type_ids)

    # Same dict-based collator as real loaders — model forward path is identical
    def _collate(batch):
        ii, am, tt = zip(*batch)
        return {"input_ids":      torch.stack(ii),
                "attention_mask": torch.stack(am),
                "token_type_ids": torch.stack(tt)}

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)


# ══════════════════════════════════════════════════════════════════════════════
# Backend alignment check  (correctness gate)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_logits(outputs):
    """Extract logit tensor from HuggingFace ModelOutput or plain tuple/tensor."""
    if hasattr(outputs, "logits") and outputs.logits is not None:
        return outputs.logits
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
        return outputs[0]
    if torch.is_tensor(outputs):
        return outputs
    raise RuntimeError(
        f"Cannot extract logits from model output of type {type(outputs)}. "
        "Expected HuggingFace ModelOutput with .logits, or a tuple/tensor.")


def _apply_backend(model, backend, has_svd_blocks: bool):
    """
    Patch model in-place to use the requested backend.

    Mirrors the backend-switch block in main() so alignment checks use
    exactly the same patching path as the production benchmark run.
    No-op for 'naive' (that is the default state after load_compressed_model).
    """
    if not has_svd_blocks or backend == "naive":
        return
    if backend == "sdpa":
        from src.encoders.backend import enable_sdpa
        enable_sdpa(model)
    elif backend == "flashsvd":
        from src.encoders.backend import enable_flashsvd
        enable_flashsvd(model)
    elif backend == "flashsvd15":
        from src.encoders.backend import enable_flashsvd15
        enable_flashsvd15(model)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


@torch.no_grad()
def run_alignment_check(model_dir, backends_to_check, device, dtype,
                        seq_len, batch_size, vocab_size, seed, warn_thresh):
    """
    Compare logits produced by each backend against the naive baseline
    on a single fixed synthetic batch (same seed → exactly reproducible).

    For each backend, loads a fresh model copy from disk, applies the backend
    patch, runs one forward pass, and computes element-wise logit differences.
    Reloading from disk (rather than deepcopy) ensures the patching path is
    identical to the production benchmark run.

    Parameters
    ----------
    model_dir : str
        Path to the compressed checkpoint (same as --model_dir in main).
    backends_to_check : list[str]
        Backends to compare against naive (do NOT include "naive" itself).
    device, dtype : str / torch.dtype
        Match the main benchmark run.
    seq_len, batch_size : int
        Sequence length / batch size for the alignment batch.
    vocab_size, seed : int
        Synthetic loader parameters (use the same as the timing run for consistency).
    warn_thresh : float
        Emit a [WARN] line when max|Δlogit| exceeds this value.

    Returns
    -------
    dict[str, dict]
        backend → {"max_abs_diff": float, "mean_abs_diff": float}
        Empty dict if backends_to_check is empty or model_dir is None.
    """
    if not model_dir or not backends_to_check:
        return {}

    from src.encoders.io import load_compressed_model

    # ── determinism guarantees ────────────────────────────────────────────
    # The following conditions hold for a valid comparison:
    #
    #   (A) @torch.no_grad() decorator on this function — no autograd state.
    #   (B) model.eval() called on EVERY loaded model — disables dropout,
    #       BatchNorm running-stats updates, and any other training-mode
    #       stochastic layers.  eval() is the single lever that matters here;
    #       no manual seed manipulation is needed.
    #   (C) Fixed synthetic batch: _make_synthetic_loader(seed=seed) always
    #       produces the same token IDs and all-1 attention mask.
    #       The batch dict is built ONCE from batch_cpu and the SAME GPU
    #       tensors are reused for every backend — no re-sampling, no
    #       re-tokenization, no hidden randomness from DataLoader shuffle.
    #   (D) logits collected as .float().detach().cpu() so BF16/FP16
    #       accumulation differences are visible rather than hidden by
    #       in-place ops.
    #
    # If max|Δlogit| is unexpectedly large, the most likely causes are:
    #   • model NOT in eval mode (dropout active) — check (B)
    #   • different batch used per backend — check (C)
    #   • a kernel bug producing numerically wrong outputs — this is what
    #     the check is designed to catch

    # One fixed synthetic batch — same seed guarantees identical tokens across all backends.
    _align_loader = _make_synthetic_loader(
        seq_len, batch_size, num_batches=2, vocab_size=vocab_size, seed=seed)
    batch_cpu = next(iter(_align_loader))   # take only the first batch; num_batches=2 is headroom

    # ── baseline: naive ────────────────────────────────────────────────────
    print(f"[align] Loading naive baseline from {model_dir} ...")
    model0, _, _ = load_compressed_model(model_dir, device=device, dtype=dtype)
    model0.eval()   # (B) disables dropout / stochastic layers

    # Detect SVD blocks so _apply_backend knows whether to patch.
    # SVD params live in layer.block (BertLayerShim wraps the SVD block),
    # NOT in layer.attention — checking l.attention.Pq always returns False.
    # Use the same signal that enable_sdpa/enable_flashsvd* relies on:
    # any module with an `attn_mode` attribute is an SVD block.
    has_svd = any(hasattr(m, "attn_mode") for m in model0.modules())

    batch = {k: v.to(device) for k, v in batch_cpu.items()}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    logits0 = _extract_logits(model0(**batch)).float().detach().cpu()
    del model0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── compare each backend ───────────────────────────────────────────────
    results = {}
    for bk in backends_to_check:
        if bk == "naive":
            continue  # baseline — skip
        if bk == "flashsvd15" and dtype == torch.float32:
            print(f"[align] SKIP  backend={bk:<12s}  flashsvd15 requires bf16/fp16, not fp32")
            continue

        model_b, _, _ = load_compressed_model(model_dir, device=device, dtype=dtype)
        model_b.eval()   # (B) same eval mode as naive baseline
        _apply_backend(model_b, bk, has_svd)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logits_b = _extract_logits(model_b(**batch)).float().detach().cpu()
        del model_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        diff     = (logits_b - logits0).abs()
        max_abs  = diff.max().item()
        mean_abs = diff.mean().item()
        results[bk] = {"max_abs_diff": max_abs, "mean_abs_diff": mean_abs}

        status = "WARN ⚠" if max_abs > warn_thresh else "OK  ✓"
        print(f"[align] {status}  backend={bk:<12s}  "
              f"max|Δlogit|={max_abs:.6f}  mean|Δlogit|={mean_abs:.6f}"
              + (f"  ← > warn_thresh={warn_thresh}" if max_abs > warn_thresh else ""))

    if results:
        all_ok = all(v["max_abs_diff"] <= warn_thresh for v in results.values())
        summary = "all backends within threshold ✓" if all_ok else "some backends exceed threshold ⚠"
        print(f"[align] {summary}  (warn_thresh={warn_thresh})")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Per-layer FLOP + traffic breakdown
# ══════════════════════════════════════════════════════════════════════════════
def _matmul_flops(M, K, N):
    """FLOPs for a [M,K] x [K,N] matmul (2×MKN)."""
    return 2 * M * K * N


def _matmul_traffic(M, K, N, ebytes):
    """Bytes for a [M,K] x [K,N] matmul (read A + read B + write C)."""
    return (M * K + K * N + M * N) * ebytes


def analyze_layer(blk, cfg, B, M, ebytes, layer_idx=0):
    """
    Compute FLOP and memory-traffic breakdown for one SVD-compressed layer.

    Returns a dict with keys:
        useful_flops, padding_flops,
        weight_traffic_bytes, act_traffic_bytes,
        per_op   (list of per-operation dicts)
    """
    dm   = cfg.hidden_size                    # 768
    H    = cfg.num_attention_heads            # 12
    dh   = dm // H                            # 64
    d_ff = cfg.intermediate_size              # 3072

    # ── rank parameters ──────────────────────────────────────────────────────
    R_attn    = blk.Pq.shape[-1]              # actual attention rank
    R_attn_p  = _next_pow2(R_attn)            # padded (kernel constraint)

    R_wo      = blk.Uo.shape[-1] if hasattr(blk, "Uo") else dm
    # WO uses plain matmul; no kernel padding constraint
    R_wo_p    = R_wo

    # FFN ranks
    has_ffn = hasattr(blk, "U1") and hasattr(blk, "V1") and hasattr(blk, "U2")
    if has_ffn:
        # V1: [R1, d_ff],  U2: [d_ff, R2],  V2: [R2, dm]
        R1    = blk.V1.shape[0]
        R2    = blk.U2.shape[1]
        R2_p  = _next_pow2(R2)
        # R1 is looped in BR1 tiles → no padding constraint for R1 itself
        R1_p  = R1
    else:
        R1 = R2 = R2_p = R1_p = 0

    ops = []

    # ── helper to record one op ───────────────────────────────────────────
    def op(name, fl_useful, fl_padded, wt_bytes, act_bytes):
        ops.append({
            "name": name,
            "useful_flops": fl_useful,
            "padding_flops": fl_padded - fl_useful,
            "weight_traffic": wt_bytes,
            "act_traffic": act_bytes,
        })

    # ── 1) QKV projections: einsum bmd,hdr->bhmr ─────────────────────────
    # Each of Q, K, V:  A=[B*M, dm]  W=[H*dm, R]  Out=[B*M*H, R]
    for qkv in ("Q_proj", "K_proj", "V_proj"):
        fl_u = _matmul_flops(B * M, dm, H * R_attn)
        fl_p = _matmul_flops(B * M, dm, H * R_attn_p)
        # weight: H * dm * R_pad; act in: B*M*dm; act out: B*H*M*R_pad
        wt = H * dm * R_attn_p * ebytes
        ac = (B * M * dm + B * H * M * R_attn_p) * ebytes
        op(qkv, fl_u, fl_p, wt, ac)

    # ── 2) Rank-space attention scores & weighted-sum ────────────────────
    # Score: [B,H,M,R] x [B,H,R,M] → [B,H,M,M]   — approximation
    # WeightedSum: [B,H,M,M] x [B,H,M,R] → [B,H,M,R]
    fl_score_u = _matmul_flops(B * H * M, R_attn,   M)
    fl_score_p = _matmul_flops(B * H * M, R_attn_p, M)
    fl_wsum_u  = _matmul_flops(B * H * M, M, R_attn)
    fl_wsum_p  = _matmul_flops(B * H * M, M, R_attn_p)
    # traffic: rank-space P tiles + score + mask (simplified)
    ac_score = (2 * B * H * M * R_attn_p + B * H * M * M) * ebytes
    op("attn_score+wsum",
       fl_score_u + fl_wsum_u,
       fl_score_p + fl_wsum_p,
       0, ac_score)

    # ── 3) Lift: [B,H,M,R] x [H,R,dh] → [B,H,M,dh]  (for V; skip Q,K) ─
    fl_lift_u = _matmul_flops(B * H * M, R_attn,   dh)
    fl_lift_p = _matmul_flops(B * H * M, R_attn_p, dh)
    wt_lift = H * R_attn_p * dh * ebytes
    ac_lift = (B * H * M * R_attn_p + B * H * M * dh) * ebytes
    op("V_lift", fl_lift_u, fl_lift_p, wt_lift, ac_lift)

    # ── 4) Output projection: Uo [dm, R_wo] + Vo [R_wo, dm] ─────────────
    # (plain matmul, no kernel padding)
    fl_uo = _matmul_flops(B * M, dm, R_wo_p)
    fl_vo = _matmul_flops(B * M, R_wo_p, dm)
    wt_wo = (dm * R_wo_p + R_wo_p * dm) * ebytes
    ac_wo = (B * M * dm + B * M * R_wo_p + B * M * dm) * ebytes
    op("WO", fl_uo + fl_vo, fl_uo + fl_vo, wt_wo, ac_wo)

    # ── 5) FFN ────────────────────────────────────────────────────────────
    if has_ffn:
        # 5a) U1: [B*M, dm] x [dm, R1] → [B*M, R1]
        fl_u1 = _matmul_flops(B * M, dm, R1_p)
        wt_u1 = dm * R1_p * ebytes
        ac_u1 = (B * M * dm + B * M * R1_p) * ebytes
        op("FFN_U1", fl_u1, fl_u1, wt_u1, ac_u1)

        # 5b) V1: [B*M, R1] x [R1, d_ff] → [B*M, d_ff]
        fl_v1 = _matmul_flops(B * M, R1_p, d_ff)
        wt_v1 = R1_p * d_ff * ebytes
        ac_v1 = (B * M * R1_p + B * M * d_ff) * ebytes
        op("FFN_V1", fl_v1, fl_v1, wt_v1, ac_v1)

        # 5c) GELU: ~8 FLOPs/element (erf approximation)
        fl_gelu = 8 * B * M * d_ff
        ac_gelu = 2 * B * M * d_ff * ebytes   # read + write
        op("GELU", fl_gelu, fl_gelu, 0, ac_gelu)

        # 5d) U2: [B*M, d_ff] x [d_ff, R2] → [B*M, R2_pad]  (padded)
        fl_u2_u = _matmul_flops(B * M, d_ff, R2)
        fl_u2_p = _matmul_flops(B * M, d_ff, R2_p)
        wt_u2 = d_ff * R2_p * ebytes
        ac_u2 = (B * M * d_ff + B * M * R2_p) * ebytes
        op("FFN_U2", fl_u2_u, fl_u2_p, wt_u2, ac_u2)

        # 5e) V2: [B*M, R2_pad] x [R2_pad, dm] → [B*M, dm]
        fl_v2_u = _matmul_flops(B * M, R2,   dm)
        fl_v2_p = _matmul_flops(B * M, R2_p, dm)
        wt_v2 = R2_p * dm * ebytes
        ac_v2 = (B * M * R2_p + B * M * dm) * ebytes
        op("FFN_V2", fl_v2_u, fl_v2_p, wt_v2, ac_v2)

    # ── aggregate ─────────────────────────────────────────────────────────
    useful  = sum(o["useful_flops"]    for o in ops)
    padding = sum(o["padding_flops"]   for o in ops)
    wt_tot  = sum(o["weight_traffic"]  for o in ops)
    ac_tot  = sum(o["act_traffic"]     for o in ops)

    return {
        "layer": layer_idx,
        "R_attn": R_attn, "R_attn_pad": R_attn_p,
        "R_wo":   R_wo,
        "R_ffn_r2": R2 if has_ffn else 0,
        "R_ffn_r2_pad": R2_p if has_ffn else 0,
        "useful_flops":   useful,
        "padding_flops":  padding,
        "total_flops":    useful + padding,
        "weight_traffic": wt_tot,
        "act_traffic":    ac_tot,
        "total_traffic":  wt_tot + ac_tot,
        "per_op": ops,
    }


def analyze_model(model, B, M, dtype):
    """Run breakdown for all encoder layers. Returns list of layer dicts + totals."""
    cfg    = model.config
    ebytes = _dtype_bytes(dtype)
    layers = _get_encoder_layers(model)

    results = []
    for i, layer in enumerate(layers):
        blk = getattr(layer, "block", None)
        if blk is None or not hasattr(blk, "Pq"):
            print(f"[warn] Layer {i} has no SVD block – skipped")
            continue
        results.append(analyze_layer(blk, cfg, B, M, ebytes, layer_idx=i))

    # totals
    totals = {
        "useful_flops":   sum(r["useful_flops"]   for r in results),
        "padding_flops":  sum(r["padding_flops"]  for r in results),
        "total_flops":    sum(r["total_flops"]     for r in results),
        "weight_traffic": sum(r["weight_traffic"]  for r in results),
        "act_traffic":    sum(r["act_traffic"]     for r in results),
        "total_traffic":  sum(r["total_traffic"]   for r in results),
    }
    totals["padding_pct"] = (
        100.0 * totals["padding_flops"] / totals["total_flops"]
        if totals["total_flops"] > 0 else 0.0
    )
    totals["arith_intensity"] = (
        totals["total_flops"] / totals["total_traffic"]
        if totals["total_traffic"] > 0 else 0.0
    )
    return results, totals


# ══════════════════════════════════════════════════════════════════════════════
# Timing benchmark  (+ optional nsys / NVTX profiling)
# ══════════════════════════════════════════════════════════════════════════════

def _nvtx_available():
    try:
        import torch.cuda.nvtx as nvtx
        return True
    except ImportError:
        return False


def print_nsys_command(script_argv: list):
    """Print the nsys command that should be used to profile this script."""
    import shlex
    args_str = " ".join(shlex.quote(a) for a in script_argv)
    print()
    print("─" * 80)
    print("  To profile with Nsight Systems, run:")
    print()
    print(f"  nsys profile \\")
    print(f"    --trace=cuda,nvtx,osrt \\")
    print(f"    --output=svd_profile \\")
    print(f"    --force-overwrite true \\")
    print(f"    python {args_str} --profile_nsys")
    print()
    print("  Then view with: nsys-ui svd_profile.nsys-rep")
    print("  Or text report: nsys stats svd_profile.nsys-rep")
    print("─" * 80)


def run_timing(model, loader, device, warmup=10, measure=50,
               profile_nsys=False, nvtx_label="inference"):
    """
    Benchmark model inference.

    Parameters
    ----------
    profile_nsys : bool
        If True, wrap each measured step with NVTX range annotations
        so that nsys captures per-step kernel timelines.
        Also calls torch.cuda.profiler.start/stop around the measure loop.
        NOTE: performance numbers measured under nsys have profiling overhead.
        Use perf_only=True (no --profile_nsys) for official latency/throughput
        numbers; use --profile_nsys only for kernel-level attribution.
    nvtx_label : str
        Base label for NVTX ranges (e.g. "inference_flashsvd15").

    Returns
    -------
    lat_ms : float
        Median batch latency in ms.
    sps : float
        Median throughput in samples/s.
    peak_mb : float
        Peak GPU memory in MB (reset before measure loop).
    avg_eff_tokens : float
        Average number of non-padding tokens per sample (from attention_mask).
        Used to compute sequence-level padding fraction.
    """
    model.eval()
    data_iter = iter(loader)

    def _next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    bs = loader.batch_size
    has_nvtx = _nvtx_available() and torch.cuda.is_available()

    # ── warmup ────────────────────────────────────────────────────────────
    with torch.no_grad():
        for _ in range(warmup):
            batch = _next_batch()
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # ── start nsys capture window ─────────────────────────────────────────
    if profile_nsys and torch.cuda.is_available():
        torch.cuda.profiler.start()
        print(f"[nsys] CUDA profiler started  ({measure} steps, label={nvtx_label!r})")

    times = []
    eff_token_sums = []   # sum of attention_mask per sample, for seq-padding stats
    with torch.no_grad():
        for step in range(measure):
            batch = _next_batch()
            batch = {k: v.to(device) for k, v in batch.items()}

            # Track effective (non-padding) tokens from the attention mask.
            # This is independent of backend and only depends on the data.
            if "attention_mask" in batch:
                eff_token_sums.append(
                    batch["attention_mask"].float().sum(dim=1).mean().item()
                )

            if has_nvtx and profile_nsys:
                torch.cuda.nvtx.range_push(f"{nvtx_label}/step{step}")

            t0 = time.perf_counter()
            model(**batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

            if has_nvtx and profile_nsys:
                torch.cuda.nvtx.range_pop()

    # ── end nsys capture window ───────────────────────────────────────────
    if profile_nsys and torch.cuda.is_available():
        torch.cuda.profiler.stop()
        print("[nsys] CUDA profiler stopped")

    peak_mb = (torch.cuda.max_memory_allocated() / 1024**2
               if torch.cuda.is_available() else 0.0)

    lat_ms = statistics.median(times) * 1000
    sps    = bs / statistics.median(times)
    avg_eff_tokens = statistics.mean(eff_token_sums) if eff_token_sums else 0.0
    return lat_ms, sps, peak_mb, avg_eff_tokens


def run_timing_repeated(model, loader, device, warmup=10, measure=50, repeat=1,
                        profile_nsys=False, nvtx_label="inference"):
    """
    Run run_timing `repeat` times independently (each with full warmup).

    Returns mean±std across repeat runs.  When repeat=1 behaves identically
    to run_timing() with std=0.

    Returns
    -------
    lat_ms      : float  — mean batch latency (ms)
    lat_ms_std  : float  — std  batch latency (ms); 0.0 when repeat == 1
    sps         : float  — mean throughput (samples/s)
    sps_std     : float  — std  throughput (samples/s); 0.0 when repeat == 1
    peak_mb     : float  — max peak memory across runs (MB)
    avg_eff_tokens : float — mean effective tokens across runs
    """
    lats, spss, mems, effs = [], [], [], []
    for r in range(repeat):
        if repeat > 1:
            print(f"[timing] Run {r+1}/{repeat} ...")
        lat, sps, mem, eff = run_timing(
            model, loader, device, warmup, measure,
            profile_nsys=profile_nsys, nvtx_label=nvtx_label)
        lats.append(lat); spss.append(sps); mems.append(mem); effs.append(eff)

    lat_mean = statistics.mean(lats)
    lat_std  = statistics.stdev(lats) if repeat > 1 else 0.0
    sps_mean = statistics.mean(spss)
    sps_std  = statistics.stdev(spss) if repeat > 1 else 0.0
    return lat_mean, lat_std, sps_mean, sps_std, max(mems), statistics.mean(effs)


# ══════════════════════════════════════════════════════════════════════════════
# Pretty printing
# ══════════════════════════════════════════════════════════════════════════════
def _fmt_g(n): return f"{n/1e9:.2f}G"
def _fmt_t(n): return f"{n/1e12:.3f}T"
def _fmt_mb(n): return f"{n/1e6:.1f}MB"


def print_report(layer_results, totals, lat_ms, sps, peak_mb, avg_eff_tokens,
                 B, M, dtype, backend, gpu_name, gpu_tflops, gpu_bw,
                 lat_ms_std=0.0, sps_std=0.0, n_repeats=1):
    """
    Print the full compute breakdown report.

    Padding terminology
    -------------------
    Two distinct sources of "wasted" FLOPs are tracked and must NOT be confused:

    (A) Rank-alignment padding (rank_pad_*):
        Triton kernels require block sizes to be powers of 2.  When the actual
        SVD rank R is not a power of 2, the kernel pads to next_pow2(R).
        This overhead is 0% when R is already a power of 2 (e.g. R=256).
        This is a STATIC property of the model — it does not vary with input.

    (B) Sequence-level padding (seq_pad_*):
        Input sentences are padded to max_length=M with [PAD] tokens.  The
        model forward pass runs over all M positions including the padding
        positions (they are masked in attention but still cost FLOPs in the
        projection layers).  This overhead depends on the average actual
        sentence length in the dataset.
        This is a DYNAMIC property of the data — it varies by task/split.

    The TFLOP rates (useful_tflop_rate, rank_pad_tflop_rate, achieved_tflop_rate)
    are all *rates* in TFLOPs/s = FLOPs / latency.  They change across backends
    because latency changes.  Use *_flops_abs for the backend-independent
    absolute FLOPs counts.
    """
    ebytes = _dtype_bytes(dtype)
    dtype_name = {torch.float32: "fp32", torch.float16: "fp16",
                  torch.bfloat16: "bf16"}.get(dtype, str(dtype))

    print()
    print("═" * 80)
    print(f"  SVD Encoder Compute Breakdown  "
          f"({backend}, {dtype_name}, B={B}, M={M})")
    print("═" * 80)

    # ── per-layer rank summary ────────────────────────────────────────────
    print(f"\n{'Layer':>5}  {'R_attn':>7}{'→pad':>5}  "
          f"{'R_ffn_r2':>9}{'→pad':>5}  "
          f"{'Useful':>10}  {'RankPad':>10}  {'RP%':>5}  "
          f"{'Traffic':>10}  {'AI':>6}")
    print("─" * 80)
    for r in layer_results:
        total = r["total_flops"]
        pad_pct = 100.0 * r["padding_flops"] / total if total > 0 else 0.0
        ai = r["total_flops"] / r["total_traffic"] if r["total_traffic"] > 0 else 0
        print(f"{r['layer']:>5}  "
              f"{r['R_attn']:>7}{r['R_attn_pad']:>5}  "
              f"{r['R_ffn_r2']:>9}{r['R_ffn_r2_pad']:>5}  "
              f"{_fmt_g(r['useful_flops']):>10}  "
              f"{_fmt_g(r['padding_flops']):>10}  "
              f"{pad_pct:>4.1f}%  "
              f"{_fmt_mb(r['total_traffic']):>10}  "
              f"{ai:>5.1f}")

    # ── totals ────────────────────────────────────────────────────────────
    print("─" * 80)
    print(f"\n  Aggregate (all {len(layer_results)} layers, B={B}, M={M}):")
    _useful_pct = (100*(1-totals['padding_flops']/totals['total_flops'])
                   if totals['total_flops'] > 0 else 0.0)
    print(f"    Useful  FLOPs (abs)    : {_fmt_t(totals['useful_flops']):>10}  "
          f"({_useful_pct:.1f}% of total)  "
          f"[latency-independent]")
    print(f"    Rank-pad FLOPs (abs)   : {_fmt_t(totals['padding_flops']):>10}  "
          f"({totals['padding_pct']:.1f}% Triton next_pow2(R) overhead)  "
          f"[latency-independent]")
    print(f"    Total   FLOPs (abs)    : {_fmt_t(totals['total_flops']):>10}  "
          f"[latency-independent]")
    print(f"    Weight traffic         : {_fmt_mb(totals['weight_traffic']):>10}")
    print(f"    Activ  traffic         : {_fmt_mb(totals['act_traffic']):>10}")
    print(f"    Total  traffic         : {_fmt_mb(totals['total_traffic']):>10}")
    print(f"    Arith intensity        : {totals['arith_intensity']:.1f} FLOP/byte")

    # ── sequence-level padding (from actual data) ──────────────────────
    if avg_eff_tokens > 0:
        seq_pad_pct = 100.0 * (1.0 - avg_eff_tokens / M)
        print(f"\n  Sequence-level padding (data-dependent, all backends identical):")
        print(f"    Avg effective tokens/sample : {avg_eff_tokens:.1f} / {M}  "
              f"({seq_pad_pct:.1f}% padding tokens)")
        print(f"    NOTE: All backends see the same batches (shuffle=False, "
              f"padding='max_length').")
        print(f"          This overhead is NOT included in the FLOPs above "
              f"(which use fixed M={M}).")
        print(f"          seq_pad_pct quantifies how much compute is 'wasted' "
              f"on [PAD] tokens.")

    # ── timing ────────────────────────────────────────────────────────────
    lat_s = lat_ms / 1000.0
    # TFLOP rates: same FLOPs / different latency → rate changes across backends
    useful_tflop_rate    = totals["useful_flops"]   / lat_s / 1e12
    rank_pad_tflop_rate  = totals["padding_flops"]  / lat_s / 1e12
    achieved_tflop_rate  = totals["total_flops"]    / lat_s / 1e12
    achieved_bw_gbs      = totals["total_traffic"]  / lat_s / 1e9

    print(f"\n  Timing:")
    if n_repeats > 1:
        # std is run-level: each run = full warmup + measure steps → 1 median → n_repeats medians
        # This is NOT step-level std within a single run.
        print(f"    Latency                : {lat_ms:.2f} ± {lat_ms_std:.2f} ms/batch"
              f"  (mean±std over {n_repeats} independent runs,")
        print(f"                             each run = {n_repeats}×[warmup+measure steps] → median)")
        print(f"    Throughput             : {sps:.1f} ± {sps_std:.1f} samples/s")
    else:
        print(f"    Latency                : {lat_ms:.2f} ms/batch  (median over measure steps)")
        print(f"    Throughput             : {sps:.1f} samples/s")
    print(f"    Peak Mem               : {peak_mb:.1f} MB")
    print(f"    Timing scope           : GPU forward only  (H2D transfer excluded)")

    print(f"\n  Achieved TFLOP rates (FLOPs/latency — change with backend speed):")
    print(f"    Useful TFLOP rate      : {useful_tflop_rate:.1f} TFLOPs/s  "
          f"  [= useful_flops / latency]")
    print(f"    Rank-pad TFLOP rate    : {rank_pad_tflop_rate:.1f} TFLOPs/s  "
          f"  [= rank_pad_flops / latency]")
    print(f"    Achieved TFLOP rate    : {achieved_tflop_rate:.1f} TFLOPs/s  "
          f"  [= total_flops / latency]")
    if gpu_tflops:
        mfu = achieved_tflop_rate / gpu_tflops
        print(f"    GPU peak               : {gpu_tflops:.0f} TFLOPs/s  ({gpu_name})")
        print(f"    MFU                    : {mfu:.1%}")
    else:
        print(f"    GPU peak               : unknown")
    print(f"    Achieved BW            : {achieved_bw_gbs:.0f} GB/s")
    if gpu_bw:
        bwu = achieved_bw_gbs / gpu_bw
        print(f"    GPU peak BW            : {gpu_bw:.0f} GB/s")
        print(f"    BW utilization         : {bwu:.1%}")

    # ── roofline ─────────────────────────────────────────────────────────
    if gpu_tflops and gpu_bw:
        ridge = (gpu_tflops * 1e12) / (gpu_bw * 1e9)  # FLOPs/byte
        ai    = totals["arith_intensity"]
        bound = "compute-bound ✓" if ai >= ridge else "memory-bound ⚠"
        print(f"\n  Roofline:")
        print(f"    Ridge point            : {ridge:.0f} FLOP/byte")
        print(f"    Arith intensity        : {ai:.1f} FLOP/byte  → {bound}")

    # ── per-op breakdown for first layer ─────────────────────────────────
    if layer_results:
        print(f"\n  Per-op breakdown (layer 0)  [rank-alignment padding only]:")
        ops = layer_results[0]["per_op"]
        print(f"    {'Op':20s}  {'Useful':>10}  {'RankPad':>10}  {'RP%':>5}  {'Traffic':>10}")
        print("    " + "─" * 60)
        for o in ops:
            total_op = o["useful_flops"] + o["padding_flops"]
            pp = 100.0 * o["padding_flops"] / total_op if total_op > 0 else 0.0
            traf = o["weight_traffic"] + o["act_traffic"]
            print(f"    {o['name']:20s}  "
                  f"{_fmt_g(o['useful_flops']):>10}  "
                  f"{_fmt_g(o['padding_flops']):>10}  "
                  f"{pp:>4.1f}%  "
                  f"{_fmt_mb(traf):>10}")

    print("\n" + "═" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="SVD BERT compute breakdown analysis")

    # model source (mutually exclusive: saved checkpoint vs on-the-fly compress)
    p.add_argument("--model_dir", default=None,
                   help="Path to saved compressed model directory "
                        "(e.g. compressed_models/bert/sst2/svd_r256_naive). "
                        "If given, skips compression.")
    p.add_argument("--method",    default="svd",
                   choices=["svd", "fwsvd", "drone", "adasvd", "dense"])
    p.add_argument("--rank",      type=int, default=None,
                   help="Unified rank for all components (fallback when --rank_attn/ffn/wo not set)")
    p.add_argument("--rank_attn", type=int, default=None,
                   help="Rank for Q/K/V attention (per-head). E.g. 48 for ra48 config.")
    p.add_argument("--rank_ffn",  type=int, default=None,
                   help="Rank for FFN intermediate projections. E.g. 256.")
    p.add_argument("--rank_wo",   type=int, default=None,
                   help="Rank for attention output projection. E.g. 208.")
    p.add_argument("--qkv_mode",  default="per_head", choices=["per_head", "full"],
                   help="QKV factorisation mode (per_head or full).")
    p.add_argument("--budget",    type=float, default=0.5)

    p.add_argument("--task",      default="sst2")
    p.add_argument("--model_id",  default=None,
                   help="HuggingFace model ID (auto-detected from compression_info.json "
                        "when --model_dir is given)")
    p.add_argument("--backend",   choices=["naive", "sdpa", "flashsvd", "flashsvd15"],
                   default="flashsvd15")
    p.add_argument("--dtype",     choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--seq_len",   type=int, default=512)
    p.add_argument("--batch_size",type=int, default=32)
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--warmup",    type=int, default=10)
    p.add_argument("--measure",   type=int, default=50)
    p.add_argument("--repeat",    type=int, default=1,
                   help="Repeat the full timing measurement N times (each with full warmup). "
                        "Report mean±std across runs. Default 1 = single run (backward compatible). "
                        "Use 3 or 5 for publishable benchmarks.")
    p.add_argument("--input_mode", choices=["real", "synthetic"], default="real",
                   help="Input distribution for timing. "
                        "'real' (default): task validation set — includes natural seq padding. "
                        "'synthetic': random tokens + all-1 attention mask — zero seq padding, "
                        "fully-utilized input. Use both modes to disentangle backend vs data effects.")
    p.add_argument("--synthetic_vocab_size", type=int, default=30522,
                   help="Vocab size for synthetic loader (default 30522, BERT). "
                        "Token IDs sampled from [100, vocab_size-100] to avoid special tokens. "
                        "Performance is insensitive to this value; set for reproducibility.")
    p.add_argument("--synthetic_seed", type=int, default=42,
                   help="RNG seed for synthetic token generation (default 42). "
                        "Performance is insensitive to specific token values; "
                        "fixing the seed ensures exact reproducibility.")
    p.add_argument("--out_csv",   default=None,
                   help="Optional CSV to append one summary row")
    p.add_argument("--profile_nsys", action="store_true",
                   help="Enable NVTX annotations + CUDA profiler start/stop "
                        "for nsys capture. Run the script under "
                        "'nsys profile --trace=cuda,nvtx ...' to capture.")
    p.add_argument("--print_nsys_cmd", action="store_true",
                   help="Print the recommended nsys command and exit.")

    # ── alignment check (correctness gate) ───────────────────────────────
    p.add_argument("--check_alignment", action="store_true",
                   help="Compare logits of the requested --backend against the naive "
                        "baseline on a single fixed synthetic batch. "
                        "Loads a fresh model copy per backend checked — adds startup time "
                        "but does not affect timing numbers. "
                        "Writes logit_max_diff / logit_mean_abs_diff to CSV.")
    p.add_argument("--align_backends", type=str, default=None,
                   help="Comma-separated backends to check against naive. "
                        "Default (not set): only the current --backend is checked. "
                        "When set explicitly, the current --backend is automatically excluded "
                        "(each job checks OTHER backends; the current one is implicit). "
                        "To check all backends in a single shot: "
                        "--backend naive --check_alignment --align_backends sdpa,flashsvd,flashsvd15")
    p.add_argument("--align_warn_thresh", type=float, default=None,
                   help="Warn when max|Δlogit| exceeds this value. "
                        "Default: dtype-dependent auto-selection "
                        "(bf16=0.05, fp16=0.02, fp32=1e-3). "
                        "Override with an explicit float to use the same threshold for all dtypes.")

    return p.parse_args()


def main():
    args = parse_args()

    # ── nsys command helper ───────────────────────────────────────────────
    if args.print_nsys_cmd:
        print_nsys_command(sys.argv[1:])
        return

    DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = DTYPE_MAP[args.dtype]
    gpu_tflops, gpu_bw, gpu_name = _gpu_specs()

    # ── load model ────────────────────────────────────────────────────────
    comp_info = {}
    if args.model_dir:
        print(f"[load] Loading pre-compressed model: {args.model_dir}")
        from src.encoders.io import load_compressed_model
        model, tokenizer, comp_info = load_compressed_model(
            args.model_dir, device=args.device, dtype=dtype)
        model_id = comp_info.get("model_id") or args.model_id or "unknown"
        # infer method from comp_info if not overridden on CLI
        if args.method == "svd":   # default value = not explicitly set
            args.method = comp_info.get("method", args.method)
    else:
        from src.encoders.compress import load_model
        _TASK_MODELS = {
            "cola":  "textattack/bert-base-uncased-CoLA",
            "sst2":  "textattack/bert-base-uncased-SST-2",
            "mrpc":  "textattack/bert-base-uncased-MRPC",
            "qqp":   "textattack/bert-base-uncased-QQP",
            "mnli":  "textattack/bert-base-uncased-MNLI",
            "qnli":  "textattack/bert-base-uncased-QNLI",
            "rte":   "textattack/bert-base-uncased-RTE",
            "stsb":  "textattack/bert-base-uncased-STS-B",
        }
        model_id = args.model_id or _TASK_MODELS.get(args.task, "bert-base-uncased")
        model, tokenizer = load_model(model_id, args.task, args.dtype, args.device)

        if args.method == "dense":
            # Dense baseline: no compression, measure as-is
            print(f"[load] Dense baseline: {model_id}")
        else:
            # on-the-fly compression (slow path)
            print(f"[compress] Compressing {args.method} rank={args.rank} ...")
            from src.encoders.compress import TASK_CFG, compress_model, prepare_loader
            loader_tmp = prepare_loader(
                args.task, tokenizer, args.seq_len, 4, split="train")
            model = compress_model(model, args.method, args.rank, args.budget,
                                   "qkv+ffn", loader_tmp, args.device, 4,
                                   rank_attn=args.rank_attn,
                                   rank_ffn=args.rank_ffn,
                                   rank_wo=args.rank_wo,
                                   qkv_mode=args.qkv_mode)

    model.eval()

    # ── build rank_config string for CSV provenance ───────────────────────
    # Encodes the compression shape so CSV rows are self-describing
    # without needing to trace back to the checkpoint directory.
    if comp_info:
        _budget = comp_info.get("budget")
        _qkv    = comp_info.get("qkv_mode", args.qkv_mode)
        if _budget:
            rank_config = f"b{_budget}_{_qkv}"
        else:
            _ra = comp_info.get("rank_attn")
            _rf = comp_info.get("rank_ffn")
            _rw = comp_info.get("rank_wo")
            # Fallback 1: CLI args (expB/expC pass --rank_attn/ffn/wo explicitly)
            if _ra is None and args.rank_attn:
                _ra, _rf, _rw = args.rank_attn, args.rank_ffn, args.rank_wo
            # Fallback 2: parse from model_dir name (e.g. svd_ra48_rf256_rw208_per_head_naive)
            if _ra is None and args.model_dir:
                import re as _re
                _m = _re.search(r'ra(\d+)_rf(\d+)_rw(\d+)',
                                os.path.basename(args.model_dir.rstrip('/')))
                if _m:
                    _ra, _rf, _rw = int(_m.group(1)), int(_m.group(2)), int(_m.group(3))
            # Fallback 3: unified rank field (old-style checkpoints)
            if _ra is None:
                _r = comp_info.get("rank") or args.rank or "?"
                _ra = _rf = _rw = _r
            rank_config = f"ra{_ra}_rf{_rf}_rw{_rw}_{_qkv}"
    elif args.method == "dense":
        rank_config = "dense"
    elif args.method == "adasvd":
        rank_config = f"b{args.budget}_{args.qkv_mode}"
    elif args.rank_attn:
        rank_config = f"ra{args.rank_attn}_rf{args.rank_ffn}_rw{args.rank_wo}_{args.qkv_mode}"
    elif args.rank:
        rank_config = f"r{args.rank}_{args.qkv_mode}"
    else:
        rank_config = "unknown"

    # ── static FLOP / traffic analysis (BEFORE backend swap) ─────────────
    # Must run on NaiveSVDBlock so Pq.shape[-1] == rank (not head_dim).
    # After enable_flashsvd15, Pq becomes [H,R,dh] and shape[-1]=dh=64,
    # which would give _next_pow2(64)=64 → 0% attention padding (wrong).
    print(f"[analyze] Computing FLOP and traffic breakdown ...")
    layer_results, totals = analyze_model(
        model, args.batch_size, args.seq_len, dtype)

    # ── alignment check (correctness gate, optional) ─────────────────────
    # Load fresh model copies per backend — does NOT affect the main timing run.
    # Runs BEFORE the backend switch so the main model stays in naive state
    # during the check (the fresh copies are loaded independently).
    align_results = {}
    if args.check_alignment:
        if not args.model_dir:
            print("[align] WARN: --check_alignment requires --model_dir; skipping.")
        else:
            if args.align_backends:
                # Explicit list: remove "naive" AND the current benchmark backend.
                # Each job checks OTHER backends; the current backend is already being
                # benchmarked by this job — no need to load it again for alignment.
                # To check all backends in one shot: --backend naive --align_backends sdpa,...
                _align_bks = [b.strip() for b in args.align_backends.split(",")
                              if b.strip() and b.strip() not in ("naive", args.backend)]
            else:
                # Default: check only the current backend vs naive baseline.
                _align_bks = [args.backend] if args.backend != "naive" else []
            if not _align_bks:
                if args.backend == "naive":
                    print("[align] backend='naive' — it IS the baseline; diff=0 by definition.")
                else:
                    print(f"[align] nothing to check "
                          f"(all specified backends excluded or current backend='{args.backend}' filtered).")
            else:
                # Resolve warn threshold: explicit override > dtype-based default
                _align_thresh = (
                    args.align_warn_thresh
                    if args.align_warn_thresh is not None
                    else _ALIGN_WARN_THRESH_BY_DTYPE.get(args.dtype, 0.05)
                )
                if args.align_warn_thresh is None:
                    print(f"[align] warn_thresh={_align_thresh} "
                          f"(auto, dtype={args.dtype}; override with --align_warn_thresh)")
                align_results = run_alignment_check(
                    args.model_dir, _align_bks, args.device, dtype,
                    args.seq_len, args.batch_size,
                    args.synthetic_vocab_size, args.synthetic_seed,
                    _align_thresh)

    # ── apply backend ─────────────────────────────────────────────────────
    # Dense models have no SVD blocks; skip backend swap silently.
    if not layer_results and args.backend not in ("naive", "sdpa"):
        print(f"[backend] No SVD blocks found — skipping {args.backend} swap (dense model)")
    if layer_results and args.backend == "sdpa":
        from src.encoders.backend import enable_sdpa
        enable_sdpa(model)
    elif layer_results and args.backend == "flashsvd":
        from src.encoders.backend import enable_flashsvd
        enable_flashsvd(model)
    elif layer_results and args.backend == "flashsvd15":
        from src.encoders.backend import enable_flashsvd15
        enable_flashsvd15(model)
    print(f"[backend] {args.backend} enabled")

    # ── build loader (real dataset or synthetic fully-padded input) ──────
    if args.input_mode == "synthetic":
        # Synthetic loader: random tokens, all-1 attention mask (0% seq padding).
        # Isolates backend kernel performance from dataset-specific padding overhead.
        # num_batches must cover warmup + measure steps (× repeat for repeated runs).
        num_batches = (args.warmup + args.measure) * args.repeat + 8   # +8 headroom
        loader = _make_synthetic_loader(
            args.seq_len, args.batch_size, num_batches,
            vocab_size=args.synthetic_vocab_size, seed=args.synthetic_seed)
        print(f"[input] synthetic: random tokens, all-1 mask, 0% seq-padding "
              f"(vocab_size={args.synthetic_vocab_size}, seed={args.synthetic_seed}, "
              f"N={num_batches * args.batch_size} pre-generated samples)")
    else:
        from src.encoders.compress import prepare_loader
        loader = prepare_loader(
            args.task, tokenizer, args.seq_len, args.batch_size,
            split="validation")
        print(f"[input] real: {args.task} validation set (natural seq-padding distribution)")

    # ── timing (+ optional nsys NVTX) ────────────────────────────────────
    if args.profile_nsys:
        print(f"[nsys] NVTX profiling enabled — wrapping each step with range annotations")
        print(f"[nsys] NOTE: latency/throughput measured under nsys have profiling overhead.")
        print(f"[nsys]       Use these numbers only for kernel attribution, not perf claims.")
        print_nsys_command(sys.argv[1:])
    print(f"[timing] Warmup={args.warmup} Measure={args.measure} steps "
          f"× Repeat={args.repeat} ...")
    lat_ms, lat_ms_std, sps, sps_std, peak_mb, avg_eff_tokens = run_timing_repeated(
        model, loader, args.device, args.warmup, args.measure, args.repeat,
        profile_nsys=args.profile_nsys,
        nvtx_label=f"inference_{args.backend}_{args.dtype}")

    # ── print report ──────────────────────────────────────────────────────
    if not layer_results:
        print("[warn] No SVD blocks found — FLOPs breakdown unavailable for this model.")
        print(f"  Latency : {lat_ms:.2f} ms/batch  |  Throughput : {sps:.1f} sps  "
              f"|  Peak Mem : {peak_mb:.1f} MB")
    else:
        print_report(layer_results, totals, lat_ms, sps, peak_mb, avg_eff_tokens,
                     args.batch_size, args.seq_len, dtype,
                     args.backend, gpu_name, gpu_tflops, gpu_bw,
                     lat_ms_std=lat_ms_std, sps_std=sps_std, n_repeats=args.repeat)

    # ── optional CSV output ───────────────────────────────────────────────
    if args.out_csv:
        import csv
        lat_s = lat_ms / 1000.0
        seq_pad_pct = 100.0 * (1.0 - avg_eff_tokens / args.seq_len) if avg_eff_tokens > 0 else 0.0
        # TFLOP rates: FLOPs / latency — these change across backends because
        # latency changes, NOT because FLOPs change.  Use *_flops_abs for the
        # latency-independent absolute FLOPs.
        useful_tflop_rate   = totals["useful_flops"]  / lat_s / 1e12
        rank_pad_tflop_rate = totals["padding_flops"] / lat_s / 1e12
        achieved_tflop_rate = totals["total_flops"]   / lat_s / 1e12
        import datetime
        row = {
            # ── provenance (who ran what, when, on which checkpoint) ──────
            "timestamp":            datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "model_id":             model_id,
            "rank_config":          rank_config,  # e.g. ra48_rf256_rw208_per_head or b0.527_per_head
            "git_commit":           _git_commit(),
            # ── experiment axes ───────────────────────────────────────────
            "task":                 args.task,
            "method":               args.method,
            "backend":              args.backend,
            "dtype":                args.dtype,
            "seq_len":              args.seq_len,
            "batch_size":           args.batch_size,
            # ── input distribution ────────────────────────────────────────
            "input_mode":           args.input_mode,   # real | synthetic
            "synthetic_seed":       str(args.synthetic_seed) if args.input_mode == "synthetic" else "",
            "synthetic_vocab_size": str(args.synthetic_vocab_size) if args.input_mode == "synthetic" else "",
            # ── absolute FLOPs (latency-independent; same across backends) ──
            "useful_flops_abs":     f"{totals['useful_flops']/1e12:.4f}",   # TFLOPs
            "rank_pad_flops_abs":   f"{totals['padding_flops']/1e12:.4f}",  # TFLOPs
            "total_flops_abs":      f"{totals['total_flops']/1e12:.4f}",    # TFLOPs
            # ── TFLOP rates (FLOPs/s; change with backend speed) ──────────
            "useful_tflop_rate":    f"{useful_tflop_rate:.3f}",
            "rank_pad_tflop_rate":  f"{rank_pad_tflop_rate:.3f}",
            "achieved_tflop_rate":  f"{achieved_tflop_rate:.3f}",
            # ── padding fractions ─────────────────────────────────────────
            "rank_pad_pct":         f"{totals['padding_pct']:.2f}",   # Triton next_pow2(R)
            "seq_pad_pct":          f"{seq_pad_pct:.2f}",              # sentence padding (0 if synthetic)
            "avg_eff_tokens":       f"{avg_eff_tokens:.1f}",
            # ── roofline ──────────────────────────────────────────────────
            "arith_intensity":      f"{totals['arith_intensity']:.1f}",
            # ── timing / memory (mean±std across repeat runs) ────────────
            "latency_ms":           f"{lat_ms:.2f}",
            "latency_ms_std":       f"{lat_ms_std:.2f}" if args.repeat > 1 else "",
            "throughput_sps":       f"{sps:.1f}",
            "throughput_sps_std":   f"{sps_std:.1f}" if args.repeat > 1 else "",
            "n_repeats":            str(args.repeat),
            "peak_mem_mb":          f"{peak_mb:.1f}",
            "mfu":                  f"{achieved_tflop_rate/gpu_tflops:.4f}" if gpu_tflops else "",
        }
        # ── alignment check (only when --check_alignment; never added to expB.csv) ──
        if args.check_alignment:
            row["logit_max_diff"] = (
                "0.000000000000e+00" if args.backend == "naive"
                else f"{align_results[args.backend]['max_abs_diff']:.12e}"
                     if args.backend in align_results
                else ""
            )
            row["logit_mean_abs_diff"] = (
                "0.000000000000e+00" if args.backend == "naive"
                else f"{align_results[args.backend]['mean_abs_diff']:.12e}"
                     if args.backend in align_results
                else ""
            )
        write_header = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"[csv] Row appended → {args.out_csv}")


if __name__ == "__main__":
    main()
