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
python eval_encoder/scripts/analyze_compute.py \
    --model_dir eval_encoder/models/sst2/svd_r256_naive \
    --task sst2 --backend flashsvd15 --dtype bf16

# Or compress on-the-fly (slower):
python eval_encoder/scripts/analyze_compute.py \
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


def _get_encoder_layers(model):
    if hasattr(model, "bert"):
        return model.bert.encoder.layer
    if hasattr(model, "roberta"):
        return model.roberta.encoder.layer
    raise RuntimeError("Unsupported architecture (no bert/roberta attribute)")


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


# ══════════════════════════════════════════════════════════════════════════════
# Pretty printing
# ══════════════════════════════════════════════════════════════════════════════
def _fmt_g(n): return f"{n/1e9:.2f}G"
def _fmt_t(n): return f"{n/1e12:.3f}T"
def _fmt_mb(n): return f"{n/1e6:.1f}MB"


def print_report(layer_results, totals, lat_ms, sps, peak_mb, avg_eff_tokens,
                 B, M, dtype, backend, gpu_name, gpu_tflops, gpu_bw):
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

    print(f"\n  Timing (median over measure steps):")
    print(f"    Latency                : {lat_ms:.2f} ms/batch")
    print(f"    Throughput             : {sps:.1f} samples/s")
    print(f"    Peak Mem               : {peak_mb:.1f} MB")

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
                        "(e.g. eval_encoder/models/sst2/svd_r256_naive). "
                        "If given, skips compression.")
    p.add_argument("--method",    default="svd",
                   choices=["svd", "fwsvd", "drone", "adasvd"])
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
    p.add_argument("--out_csv",   default=None,
                   help="Optional CSV to append one summary row")
    p.add_argument("--profile_nsys", action="store_true",
                   help="Enable NVTX annotations + CUDA profiler start/stop "
                        "for nsys capture. Run the script under "
                        "'nsys profile --trace=cuda,nvtx ...' to capture.")
    p.add_argument("--print_nsys_cmd", action="store_true",
                   help="Print the recommended nsys command and exit.")
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
    if args.model_dir:
        print(f"[load] Loading pre-compressed model: {args.model_dir}")
        from eval_encoder.load_compressed_model import load_compressed_model
        model, tokenizer, comp_info = load_compressed_model(
            args.model_dir, device=args.device, dtype=dtype)
        model_id = comp_info.get("model_id") or args.model_id or "unknown"
        # infer method from comp_info if not overridden on CLI
        if args.method == "svd":   # default value = not explicitly set
            args.method = comp_info.get("method", args.method)
    else:
        # on-the-fly compression (slow path)
        print(f"[compress] Compressing {args.method} rank={args.rank} ...")
        from eval_encoder.run_encoder_benchmark import (
            load_model, TASK_CFG, compress_model)
        if args.model_id is None:
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
            model_id = _TASK_MODELS.get(args.task, "bert-base-uncased")
        else:
            model_id = args.model_id
        model, tokenizer = load_model(model_id, args.task, args.dtype, args.device)

        # build a minimal calibration loader
        from eval_encoder.run_encoder_benchmark import prepare_loader
        loader_tmp = prepare_loader(
            args.task, tokenizer, args.seq_len, 4, split="train")
        model = compress_model(model, args.method, args.rank, args.budget,
                               "qkv+ffn", loader_tmp, args.device, 4,
                               rank_attn=args.rank_attn,
                               rank_ffn=args.rank_ffn,
                               rank_wo=args.rank_wo,
                               qkv_mode=args.qkv_mode)

    model.eval()

    # ── static FLOP / traffic analysis (BEFORE backend swap) ─────────────
    # Must run on NaiveSVDBlock so Pq.shape[-1] == rank (not head_dim).
    # After enable_flashsvd15, Pq becomes [H,R,dh] and shape[-1]=dh=64,
    # which would give _next_pow2(64)=64 → 0% attention padding (wrong).
    print(f"[analyze] Computing FLOP and traffic breakdown ...")
    layer_results, totals = analyze_model(
        model, args.batch_size, args.seq_len, dtype)

    # ── apply backend ─────────────────────────────────────────────────────
    # Dense models have no SVD blocks; skip backend swap silently.
    if not layer_results and args.backend not in ("naive", "sdpa"):
        print(f"[backend] No SVD blocks found — skipping {args.backend} swap (dense model)")
    if layer_results and args.backend == "sdpa":
        from eval_encoder.flashsvd_backend import enable_sdpa
        enable_sdpa(model)
    elif layer_results and args.backend == "flashsvd":
        from eval_encoder.flashsvd_backend import enable_flashsvd
        enable_flashsvd(model)
    elif layer_results and args.backend == "flashsvd15":
        from eval_encoder.flashsvd_backend import enable_flashsvd15
        enable_flashsvd15(model)
    print(f"[backend] {args.backend} enabled")

    # ── build validation loader ───────────────────────────────────────────
    from eval_encoder.run_encoder_benchmark import prepare_loader
    loader = prepare_loader(
        args.task, tokenizer, args.seq_len, args.batch_size,
        split="validation")

    # ── timing (+ optional nsys NVTX) ────────────────────────────────────
    if args.profile_nsys:
        print(f"[nsys] NVTX profiling enabled — wrapping each step with range annotations")
        print(f"[nsys] NOTE: latency/throughput measured under nsys have profiling overhead.")
        print(f"[nsys]       Use these numbers only for kernel attribution, not perf claims.")
        print_nsys_command(sys.argv[1:])
    print(f"[timing] Warmup={args.warmup} Measure={args.measure} steps ...")
    try:
        lat_ms, sps, peak_mb, avg_eff_tokens = run_timing(
            model, loader, args.device, args.warmup, args.measure,
            profile_nsys=args.profile_nsys,
            nvtx_label=f"inference_{args.backend}_{args.dtype}")
    except Exception as e:
        print(f"[skip] run_timing failed ({type(e).__name__}): {e}")
        sys.exit(0)

    # ── print report ──────────────────────────────────────────────────────
    if not layer_results:
        print("[warn] No SVD blocks found — FLOPs breakdown unavailable for this model.")
        print(f"  Latency : {lat_ms:.2f} ms/batch  |  Throughput : {sps:.1f} sps  "
              f"|  Peak Mem : {peak_mb:.1f} MB")
    else:
        print_report(layer_results, totals, lat_ms, sps, peak_mb, avg_eff_tokens,
                     args.batch_size, args.seq_len, dtype,
                     args.backend, gpu_name, gpu_tflops, gpu_bw)

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
        row = {
            "task":                 args.task,
            "method":               args.method,
            "backend":              args.backend,
            "dtype":                args.dtype,
            "seq_len":              args.seq_len,
            "batch_size":           args.batch_size,
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
            "seq_pad_pct":          f"{seq_pad_pct:.2f}",              # sentence padding
            "avg_eff_tokens":       f"{avg_eff_tokens:.1f}",
            # ── roofline ──────────────────────────────────────────────────
            "arith_intensity":      f"{totals['arith_intensity']:.1f}",
            # ── timing / memory ───────────────────────────────────────────
            "latency_ms":           f"{lat_ms:.2f}",
            "throughput_sps":       f"{sps:.1f}",
            "peak_mem_mb":          f"{peak_mb:.1f}",
            "mfu":                  f"{achieved_tflop_rate/gpu_tflops:.4f}" if gpu_tflops else "",
        }
        write_header = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"[csv] Row appended → {args.out_csv}")


if __name__ == "__main__":
    main()
