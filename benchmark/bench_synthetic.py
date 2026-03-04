#!/usr/bin/env python3
"""
bench_synthetic.py — Batch-size scaling microbenchmark
=======================================================
Measures peak_mem_infer_mb and throughput_sps using synthetic random inputs.
No GLUE datasets, no dataloader overhead — pure kernel measurement.

Sweep:
  seq_len = 512 (fixed)
  bs ∈ {8, 16, 32, 64}
  backends: fp32 Naive(einsum) / Naive(SDPA) / FlashSVD
  optional: bf16 FlashSVD  (--include_bf16)

SVD: per-head ra48/rf256/rw208, scope=qkv+ffn

Output CSV:
  eval_encoder/eval_results/batch_scaling.csv

Figures:
  eval_encoder/eval_results/figures/batch_memory.{png,pdf}
  eval_encoder/eval_results/figures/batch_throughput.{png,pdf}

Usage:
  python eval_encoder/bench_synthetic.py
  python eval_encoder/bench_synthetic.py --bs 8,16,32,64 --include_bf16
  python eval_encoder/bench_synthetic.py --plot_only
"""

import os, sys, csv, gc, copy, time, argparse
from typing import List

import torch
import torch.nn as nn

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── project imports ────────────────────────────────────────────────────────────
from transformers import AutoModelForSequenceClassification

from eval_encoder.blocks import NaiveSVDBlock, BertLayerShim
from eval_encoder.flashsvd_backend import enable_flashsvd

# ── constants ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "textattack/bert-base-uncased-SST-2"
BERT_VOCAB_SIZE = 30522
RANK_ATTN       = 48
RANK_FFN        = 256
RANK_WO         = 208
WARMUP_STEPS    = 20
MEASURE_STEPS   = 50
DEFAULT_SEQ     = 512
DEFAULT_BS_LIST = [8, 16, 32, 64]

OUT_DIR  = os.path.join(_DIR, "eval_results")
FIG_DIR  = os.path.join(OUT_DIR, "figures")
CSV_PATH = os.path.join(OUT_DIR, "batch_scaling.csv")
os.makedirs(FIG_DIR, exist_ok=True)

CSV_FIELDS = [
    "model_id", "method", "qkv_mode", "rank",
    "bs", "seq_len", "dtype", "backend", "attn_mode",
    "latency_ms", "throughput_sps", "peak_mem_infer_mb",
]

DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


# ── model helpers ──────────────────────────────────────────────────────────────

def _detect_bert_layers(model):
    if hasattr(model, "bert"):
        return model.bert.encoder.layer
    if hasattr(model, "roberta"):
        return model.roberta.encoder.layer
    raise ValueError("bench_synthetic: only BERT/RoBERTa supported")


def _apply_svd_inplace(model: nn.Module, rank_attn: int, rank_ff: int,
                       rank_wo: int, attn_mode: str, device: str) -> nn.Module:
    """
    Apply per-head SVD to all encoder layers in-place.
    Uses _build_plain_svd_helpers() from run_encoder_benchmark to keep
    factorisation identical to the main benchmark.
    """
    from eval_encoder.run_encoder_benchmark import _build_plain_svd_helpers

    encoder_layers = _detect_bert_layers(model)
    per_head_fn, low_rank_fn = _build_plain_svd_helpers(model)

    for i, layer in enumerate(encoder_layers):
        blk = NaiveSVDBlock(
            layer, rank_attn, rank_ff,
            per_head_fn, low_rank_fn,
            rank_wo,
            qkv_mode="per_head",
            attn_mode=attn_mode,
        )
        encoder_layers[i] = BertLayerShim(blk).to(device).eval()

    return model


def _set_attn_mode(model: nn.Module, mode: str):
    """Flip attn_mode on all NaiveSVDBlock layers without rebuilding the model."""
    for m in model.modules():
        if isinstance(m, NaiveSVDBlock):
            m.attn_mode = mode


# ── synthetic inputs ───────────────────────────────────────────────────────────

def make_inputs(bs: int, seq_len: int, device: str) -> dict:
    return dict(
        input_ids      = torch.randint(1, BERT_VOCAB_SIZE, (bs, seq_len), device=device),
        attention_mask = torch.ones(bs, seq_len, dtype=torch.long, device=device),
    )


# ── measurement ────────────────────────────────────────────────────────────────

def measure(model: nn.Module, inputs: dict, device: str,
            warmup: int, steps: int):
    """
    Returns (latency_ms, throughput_sps, peak_mem_infer_mb).
    Peak memory is reset immediately before the measurement window.
    """
    bs = inputs["input_ids"].shape[0]
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            model(**inputs)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            model(**inputs)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    peak_mb    = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    lat_ms     = elapsed / steps * 1000
    throughput = bs * steps / elapsed

    return lat_ms, throughput, peak_mb


# ── benchmark loop ─────────────────────────────────────────────────────────────

# (backend_tag, attn_mode, dtype_str)
def _build_configs(include_bf16: bool):
    cfgs = [
        ("naive",    "einsum", "fp32"),
        ("naive",    "sdpa",   "fp32"),
        ("flashsvd", "triton", "fp32"),
    ]
    if include_bf16:
        cfgs += [
            ("naive",    "einsum", "bf16"),
            ("flashsvd", "triton", "bf16"),
        ]
    return cfgs


def run_benchmark(args) -> List[dict]:
    device  = args.device
    bs_list = [int(x) for x in args.bs.split(",")]
    configs = _build_configs(args.include_bf16)

    # ── compress once (fp32 einsum as base) ────────────────────────────────────
    print(f"Loading {args.model_id} …")
    base_model = (
        AutoModelForSequenceClassification
        .from_pretrained(args.model_id, trust_remote_code=True)
        .to(device=device, dtype=torch.float32)
        .eval()
    )
    print(f"Applying SVD (ra{RANK_ATTN}/rf{RANK_FFN}/rw{RANK_WO}) …")
    _apply_svd_inplace(base_model, RANK_ATTN, RANK_FFN, RANK_WO,
                       attn_mode="einsum", device=device)
    print("Compression done.\n")

    rows = []

    for bs in bs_list:
        inputs = make_inputs(bs, args.seq_len, device)
        print(f"=== bs={bs}  seq={args.seq_len} ===")

        for backend, attn_mode, dtype_str in configs:
            label = f"{dtype_str}/{backend}"
            print(f"  [{label:20s}] … ", end="", flush=True)

            try:
                # --- prepare variant ---
                m = copy.deepcopy(base_model)

                # dtype conversion (weights + buffers)
                pt_dtype = DTYPE_MAP[dtype_str]
                m = m.to(dtype=pt_dtype, device=device).eval()

                # backend switch
                if backend == "flashsvd":
                    enable_flashsvd(m)
                elif backend == "naive" and attn_mode == "sdpa":
                    _set_attn_mode(m, "sdpa")

                lat, thr, mem = measure(m, inputs, device,
                                        warmup=args.warmup, steps=args.steps)

                print(f"lat={lat:6.1f} ms  thr={thr:6.0f} sps  mem={mem:6.0f} MB")

                rows.append({
                    "model_id":          args.model_id,
                    "method":            "svd",
                    "qkv_mode":          "per_head",
                    "rank":              f"ra{RANK_ATTN}_rf{RANK_FFN}_rw{RANK_WO}",
                    "bs":                bs,
                    "seq_len":           args.seq_len,
                    "dtype":             dtype_str,
                    "backend":           backend,
                    "attn_mode":         attn_mode,
                    "latency_ms":        f"{lat:.2f}",
                    "throughput_sps":    f"{thr:.1f}",
                    "peak_mem_infer_mb": f"{mem:.1f}",
                })

            except torch.cuda.OutOfMemoryError:
                print("OOM")
                rows.append({
                    "model_id": args.model_id, "method": "svd",
                    "qkv_mode": "per_head",
                    "rank": f"ra{RANK_ATTN}_rf{RANK_FFN}_rw{RANK_WO}",
                    "bs": bs, "seq_len": args.seq_len,
                    "dtype": dtype_str, "backend": backend, "attn_mode": attn_mode,
                    "latency_ms": "OOM", "throughput_sps": "OOM",
                    "peak_mem_infer_mb": "OOM",
                })

            finally:
                try:
                    del m
                except NameError:
                    pass
                torch.cuda.empty_cache()
                gc.collect()

        print()

    # ── write CSV ──────────────────────────────────────────────────────────────
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {CSV_PATH}")

    return rows


# ── plotting ───────────────────────────────────────────────────────────────────

STYLE = {
    # (dtype, backend): (color, marker, linestyle, legend_label)
    ("fp32", "naive",    "einsum"): ("#D32F2F", "o", (4, 2), "fp32-einsum"),
    ("fp32", "naive",    "sdpa"):   ("#F57C00", "s", (2, 2), "fp32-SDPA"),
    ("fp32", "flashsvd", "triton"): ("#1565C0", "^", (1, 1), "fp32-Flash"),
    ("bf16", "naive",    "einsum"): ("#558B2F", "o", (4, 2), "bf16-einsum"),
    ("bf16", "flashsvd", "triton"): ("#00838F", "D", None,   "bf16-Flash"),
}


def _load_rows_numeric(rows):
    out = []
    for r in rows:
        try:
            out.append({
                "bs":    int(r["bs"]),
                "dtype": r["dtype"],
                "backend": r["backend"],
                "attn_mode": r["attn_mode"],
                "mem":  float(r["peak_mem_infer_mb"]),
                "thr":  float(r["throughput_sps"]),
                "lat":  float(r["latency_ms"]),
            })
        except (ValueError, KeyError):
            pass  # OOM or missing
    return out


def plot_results(rows: List[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from collections import defaultdict

    data  = _load_rows_numeric(rows)
    if not data:
        print("No numeric data to plot.")
        return

    # group by (dtype, backend, attn_mode)
    groups = defaultdict(lambda: {"bs": [], "mem": [], "thr": []})
    for r in data:
        key = (r["dtype"], r["backend"], r["attn_mode"])
        groups[key]["bs"].append(r["bs"])
        groups[key]["mem"].append(r["mem"])
        groups[key]["thr"].append(r["thr"])

    # sort each group by bs
    for g in groups.values():
        idx = sorted(range(len(g["bs"])), key=lambda i: g["bs"][i])
        for field in ("bs", "mem", "thr"):
            g[field] = [g[field][i] for i in idx]

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    LW, MS = 2.0, 8

    bs_vals = sorted(set(r["bs"] for r in data))
    subtitle = (f"SVD per-head (ra{RANK_ATTN}/rf{RANK_FFN}/rw{RANK_WO})  ·  "
                f"seq={DEFAULT_SEQ}  ·  microbenchmark ({MEASURE_STEPS} steps after "
                f"{WARMUP_STEPS} warmup)")

    for metric, ylabel, stem in [
        ("mem", "Peak Inference Memory (MB)",   "batch_memory"),
        ("thr", "Throughput (samples / sec)",   "batch_throughput"),
    ]:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))

        for key, vals in sorted(groups.items()):
            if key not in STYLE:
                continue
            color, marker, dashes, label = STYLE[key]
            kw = dict(color=color, marker=marker, lw=LW, ms=MS,
                      label=label, zorder=4)
            if key[2] == "einsum":          # fade einsum: reference, not hero
                kw["alpha"] = 0.50
                kw["lw"]    = LW - 0.4
            if dashes:
                kw["dashes"] = dashes
            ax.plot(vals["bs"], vals[metric], **kw)

        # annotate FlashSVD memory reduction vs einsum at each bs
        if metric == "mem":
            einsum_by_bs = {}
            for key, vals in groups.items():
                if key == ("fp32", "naive", "einsum"):
                    einsum_by_bs = dict(zip(vals["bs"], vals["mem"]))
            flash_by_bs = {}
            for key, vals in groups.items():
                if key == ("fp32", "flashsvd", "triton"):
                    flash_by_bs = dict(zip(vals["bs"], vals["mem"]))
            flash_color = STYLE[("fp32", "flashsvd", "triton")][0]
            for bs in bs_vals:
                if bs in einsum_by_bs and bs in flash_by_bs:
                    red = (einsum_by_bs[bs] - flash_by_bs[bs]) / einsum_by_bs[bs] * 100
                    mid = (einsum_by_bs[bs] + flash_by_bs[bs]) / 2
                    ax.text(bs, mid, f"−{red:.0f}%",
                            color=flash_color, fontsize=9.5, fontweight="bold",
                            va="center", ha="center",
                            bbox=dict(boxstyle="round,pad=0.15",
                                      fc="white", ec="none", alpha=0.8))

        ax.set_xticks(bs_vals)
        ax.set_xticklabels([str(b) for b in bs_vals])
        ax.set_xlim(bs_vals[0] - 4, bs_vals[-1] + 8)
        ax.set_xlabel("Batch Size", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(bottom=0)
        title = ("Memory Scaling: dtype × Backend"   if metric == "mem"
                 else "Throughput Scaling: dtype × Backend")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="upper left" if metric == "mem" else "upper right",
                  framealpha=0.9, fontsize=9)
        ax.grid(axis="y", alpha=0.25, lw=0.8)
        fig.text(0.5, -0.04, subtitle, ha="center", fontsize=8, color="#555555")
        fig.tight_layout()

        for ext in ("png", "pdf"):
            path = os.path.join(FIG_DIR, f"{stem}.{ext}")
            fig.savefig(path, dpi=180, bbox_inches="tight")
            print(f"Saved: {path}")
        plt.close(fig)


# ── print summary table ────────────────────────────────────────────────────────

def print_summary(rows: List[dict]):
    data = _load_rows_numeric(rows)
    if not data:
        return

    SEP = "=" * 90
    print()
    print(SEP)
    print(f"{'bs':>4}  {'dtype':>5}  {'backend':>10}  {'attn':>6}  "
          f"{'lat(ms)':>8}  {'thr(sps)':>9}  {'mem(MB)':>8}")
    print("-" * 90)
    for r in sorted(data, key=lambda x: (x["bs"], x["dtype"], x["backend"])):
        print(f"{r['bs']:>4}  {r['dtype']:>5}  {r['backend']:>10}  "
              f"{r['attn_mode']:>6}  {r['lat']:>8.1f}  {r['thr']:>9.0f}  "
              f"{r['mem']:>8.1f}")
    print(SEP)

    # FlashSVD vs einsum reduction summary
    einsum = {r["bs"]: r for r in data
              if r["dtype"] == "fp32" and r["backend"] == "naive"
              and r["attn_mode"] == "einsum"}
    flash  = {r["bs"]: r for r in data
              if r["dtype"] == "fp32" and r["backend"] == "flashsvd"}
    if einsum and flash:
        print("\nFp32 FlashSVD vs Naive(einsum):")
        print(f"  {'bs':>4}  {'mem_red':>8}  {'thr_spd':>8}")
        for bs in sorted(set(einsum) & set(flash)):
            m_red = (einsum[bs]["mem"] - flash[bs]["mem"]) / einsum[bs]["mem"] * 100
            t_spd = flash[bs]["thr"] / einsum[bs]["thr"]
            print(f"  {bs:>4}  {m_red:>7.1f}%  ×{t_spd:>6.2f}")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model_id",     default=DEFAULT_MODEL,
                   help="HuggingFace model ID (BERT/RoBERTa only)")
    p.add_argument("--seq_len",      type=int, default=DEFAULT_SEQ)
    p.add_argument("--bs",           default=",".join(map(str, DEFAULT_BS_LIST)),
                   help="Comma-separated batch sizes, e.g. 8,16,32,64")
    p.add_argument("--warmup",       type=int, default=WARMUP_STEPS)
    p.add_argument("--steps",        type=int, default=MEASURE_STEPS)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--include_bf16", action="store_true",
                   help="Also benchmark bf16 variants")
    p.add_argument("--plot_only",    action="store_true",
                   help="Skip benchmark, regenerate figures from existing CSV")
    return p.parse_args()


def main():
    args = parse_args()

    if args.plot_only:
        print(f"Re-plotting from {CSV_PATH} …")
        with open(CSV_PATH) as f:
            rows = list(csv.DictReader(f))
    else:
        rows = run_benchmark(args)

    print_summary(rows)
    plot_results(rows)
    print("Done.")


if __name__ == "__main__":
    main()
