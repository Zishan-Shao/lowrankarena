#!/usr/bin/env python3
"""
dtype × backend scaling figures (saved to figures/):
  figures/dtype_memory_scaling.{png,pdf}
  figures/dtype_throughput_scaling.{png,pdf}

Data:
  fp32 Naive(einsum) / Naive(SDPA) / FlashSVD  — from plot_seqlen_scaling.py
  bf16 FlashSVD                                 — encoder_runs.csv, 8 GLUE tasks avg

Note: bf16 Naive data not yet available.
      Run `bash eval_encoder/scripts/run_bf16_naive.sh` to collect it.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FIG_DIR = "eval_encoder/eval_results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# fp32 data (from benchmark logs, averaged over 8 GLUE tasks)
# ─────────────────────────────────────────────────────────────────
SEQ = np.array([128, 256, 512])

MEM_FP32 = {
    "einsum": np.array([559.0,  942.1,  2003.9]),
    "sdpa":   np.array([840.0, 1078.0,  1566.0]),
    "flash":  np.array([377.7,  484.8,   708.1]),
}
THR_FP32 = {
    "einsum": np.array([1325., 530., 195.]),
    "sdpa":   np.array([1487., 756., 352.]),
    "flash":  np.array([1460., 725., 336.]),
}

# ─────────────────────────────────────────────────────────────────
# bf16 FlashSVD data — read from CSV, average over 8 GLUE tasks
# ─────────────────────────────────────────────────────────────────
CSV_PATH = "eval_encoder/eval_results/encoder_runs.csv"

def load_bf16_flash():
    mem_by_seq = {}
    thr_by_seq = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if r.get("dtype") != "bf16":
                continue
            try:
                seq   = int(r["seq_len"])
                infer = float(r["peak_mem_infer_mb"])
                thr   = float(r["throughput_sps"])
            except (ValueError, KeyError):
                continue
            mem_by_seq.setdefault(seq, []).append(infer)
            thr_by_seq.setdefault(seq, []).append(thr)

    seqs = sorted(mem_by_seq)
    return (
        np.array(seqs),
        np.array([np.mean(mem_by_seq[s]) for s in seqs]),
        np.array([np.mean(thr_by_seq[s]) for s in seqs]),
    )

BF16_SEQ, MEM_BF16_FLASH, THR_BF16_FLASH = load_bf16_flash()

# bf16 Naive: peak_mem_mb from encoder_runs_pre_v2.csv (8 tasks identical per seq)
MEM_BF16_NAIVE = np.array([257.9, 377.0, 621.2])

# ─────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────
# fp32 palette
C_EINSUM = "#D32F2F"   # red
C_SDPA   = "#F57C00"   # orange
C_FLASH  = "#1565C0"   # blue
# bf16 palette
C_BF16_NAIVE = "#558B2F"   # olive green
C_BF16_FLASH = "#00838F"   # teal

SUBTITLE = "SVD per-head (ra48/rf256/rw208)  ·  batch=32  ·  averaged over 8 GLUE tasks"

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

LW, MS = 2.0, 8

def _setup_xax(ax):
    ax.set_xticks(SEQ)
    ax.set_xticklabels([str(s) for s in SEQ])
    ax.set_xlim(96, 576)
    ax.set_xlabel("Sequence Length", fontsize=11)

def _save(fig, stem):
    png = f"{FIG_DIR}/{stem}.png"
    pdf = f"{FIG_DIR}/{stem}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf,           bbox_inches="tight")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────
# Figure 1 – Memory Scaling: fp32 (3 backends) + bf16 (naive + flash)
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.2))

# fp32 lines (dashed — reference tier)
ax.plot(SEQ, MEM_FP32["einsum"], color=C_EINSUM, marker="o",
        lw=LW, ms=MS, dashes=(4, 2), label="fp32-einsum", zorder=3)
ax.plot(SEQ, MEM_FP32["sdpa"],   color=C_SDPA,   marker="s",
        lw=LW, ms=MS, dashes=(2, 2), label="fp32-SDPA",   zorder=3)
ax.plot(SEQ, MEM_FP32["flash"],  color=C_FLASH,  marker="^",
        lw=LW, ms=MS, dashes=(1, 1), label="fp32-Flash",  zorder=4)

# bf16 lines (solid — proposed tier)
ax.plot(SEQ, MEM_BF16_NAIVE,  color=C_BF16_NAIVE,  marker="o",
        lw=LW, ms=MS, label="bf16-Naive", zorder=4)
ax.plot(BF16_SEQ, MEM_BF16_FLASH, color=C_BF16_FLASH, marker="D",
        lw=LW + 0.6, ms=MS, label="bf16-Flash", zorder=5)

# Annotate bf16 FlashSVD reduction vs bf16 Naive
MEM_RED_BF16 = (MEM_BF16_NAIVE - MEM_BF16_FLASH) / MEM_BF16_NAIVE * 100
for sl, mn, mf, red in zip(SEQ, MEM_BF16_NAIVE, MEM_BF16_FLASH, MEM_RED_BF16):
    mid = (mn + mf) / 2
    ax.text(sl + 14, mid, f"−{red:.0f}%",
            color=C_BF16_FLASH, fontsize=8.5,
            va="center", ha="left", fontstyle="italic")

# fp32 SDPA cross-over annotation
ax.annotate(
    "fp32 SDPA > einsum\n(MEA tile overhead)",
    xy=(128, MEM_FP32["sdpa"][0]), xytext=(148, 1380),
    fontsize=7.5, color=C_SDPA,
    arrowprops=dict(arrowstyle="->", color=C_SDPA, lw=0.8),
)

ax.set_ylim(0, 2500)
ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
ax.set_ylabel("Peak GPU Memory (MB)", fontsize=11)
ax.set_title("Memory Scaling: dtype × Backend", fontweight="bold")
_setup_xax(ax)
ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
ax.grid(axis="y", alpha=0.25, lw=0.8)
fig.text(0.5, -0.04, SUBTITLE, ha="center", fontsize=8.5, color="#555555")
fig.tight_layout()
_save(fig, "dtype_memory_scaling")

# ─────────────────────────────────────────────────────────────────
# Figure 2 – Memory Reduction (%) — bf16 FlashSVD vs bf16 Naive
# Rising line showing advantage grows with seq_len
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.2))

# bf16: flash vs naive
ax.plot(SEQ, MEM_RED_BF16,
        color=C_BF16_FLASH, marker="D", lw=2.4, ms=MS, zorder=5,
        label="bf16  FlashSVD vs Naive")
ax.fill_between(SEQ, 0, MEM_RED_BF16, color=C_BF16_FLASH, alpha=0.10)

# fp32 for comparison (dashed)
MEM_RED_FP32 = (MEM_FP32["einsum"] - MEM_FP32["flash"]) / MEM_FP32["einsum"] * 100
ax.plot(SEQ, MEM_RED_FP32,
        color=C_FLASH, marker="^", lw=LW, ms=MS, dashes=(3, 2), zorder=4,
        label="fp32  FlashSVD vs Naive(einsum)")
ax.fill_between(SEQ, 0, MEM_RED_FP32, color=C_FLASH, alpha=0.06)

# Annotate values
for sl, r in zip(SEQ, MEM_RED_BF16):
    ax.text(sl, r + 1.5, f"{r:.1f}%",
            color=C_BF16_FLASH, fontsize=11,
            va="bottom", ha="center", fontweight="bold")
for sl, r in zip(SEQ, MEM_RED_FP32):
    ax.text(sl, r - 3.5, f"{r:.1f}%",
            color=C_FLASH, fontsize=9,
            va="top", ha="center")

ax.set_ylim(0, 80)
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax.set_ylabel("Peak inference memory reduction (%)", fontsize=11)
ax.set_title("FlashSVD Memory Reduction by dtype", fontweight="bold")
_setup_xax(ax)
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(axis="y", alpha=0.25, lw=0.8)
fig.text(0.5, -0.04, SUBTITLE, ha="center", fontsize=8.5, color="#555555")
fig.tight_layout()
_save(fig, "dtype_memory_reduction")

# ─────────────────────────────────────────────────────────────────
# Figure 3 – Throughput Scaling (fp32 all backends + bf16 FlashSVD)
# No bf16 naive throughput data available yet
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.2))

ax.plot(SEQ, THR_FP32["einsum"], color=C_EINSUM, marker="o",
        lw=LW, ms=MS, dashes=(4, 2), label="fp32-einsum", zorder=3)
ax.plot(SEQ, THR_FP32["sdpa"],   color=C_SDPA,   marker="s",
        lw=LW, ms=MS, dashes=(2, 2), label="fp32-SDPA",   zorder=3)
ax.plot(SEQ, THR_FP32["flash"],  color=C_FLASH,  marker="^",
        lw=LW, ms=MS, dashes=(1, 1), label="fp32-Flash",  zorder=4)
ax.plot(BF16_SEQ, THR_BF16_FLASH, color=C_BF16_FLASH, marker="D",
        lw=LW + 0.6, ms=MS, label="bf16-Flash", zorder=5)

# Annotate bf16 speedup vs fp32 FlashSVD at seq=512
sl512 = SEQ[-1]
tf_fp32 = THR_FP32["flash"][-1]
tf_bf16 = THR_BF16_FLASH[-1]
spd = tf_bf16 / tf_fp32
ax.annotate(
    f"×{spd:.1f} vs fp32",
    xy=(sl512, tf_bf16), xytext=(sl512 - 110, tf_bf16 + 280),
    color=C_BF16_FLASH, fontsize=9.5, fontstyle="italic",
    arrowprops=dict(arrowstyle="->", color=C_BF16_FLASH, lw=1.0),
)

ax.set_ylim(0, 3200)
ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
ax.set_ylabel("Throughput (samples / sec)", fontsize=11)
ax.set_title("Throughput Scaling: dtype × Backend", fontweight="bold")
_setup_xax(ax)
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
ax.grid(axis="y", alpha=0.25, lw=0.8)
fig.text(0.5, -0.04, SUBTITLE, ha="center", fontsize=8.5, color="#555555")
fig.tight_layout()
_save(fig, "dtype_throughput_scaling")

# ─────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────
SEP = "=" * 80
print()
print(SEP)
print("Memory (MB)")
print(f"  {'seq':>4}  {'fp32 einsum':>12}  {'fp32 SDPA':>10}  {'fp32 Flash':>10}  "
      f"{'bf16 Naive':>10}  {'bf16 Flash':>10}  {'bf16 red%':>9}")
print("-" * 80)
for i, sl in enumerate(SEQ):
    red = MEM_RED_BF16[i]
    print(f"  {sl:>4}  {MEM_FP32['einsum'][i]:>12.1f}  {MEM_FP32['sdpa'][i]:>10.1f}  "
          f"{MEM_FP32['flash'][i]:>10.1f}  {MEM_BF16_NAIVE[i]:>10.1f}  "
          f"{MEM_BF16_FLASH[i]:>10.1f}  {red:>8.1f}%")
print()
print("Throughput (sps) — bf16 naive not yet collected")
print(f"  {'seq':>4}  {'fp32 einsum':>12}  {'fp32 SDPA':>10}  {'fp32 Flash':>10}  {'bf16 Flash':>10}  {'speedup':>8}")
print("-" * 80)
for i, sl in enumerate(SEQ):
    tf  = THR_FP32["flash"][i]
    tbf = THR_BF16_FLASH[i]
    print(f"  {sl:>4}  {THR_FP32['einsum'][i]:>12.0f}  {THR_FP32['sdpa'][i]:>10.0f}  "
          f"{tf:>10.0f}  {tbf:>10.0f}  x{tbf/tf:>6.2f}")
print(SEP)
