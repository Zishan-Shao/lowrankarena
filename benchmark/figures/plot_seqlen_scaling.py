#!/usr/bin/env python3
"""
NeurIPS-style seq-len scaling figures (saved separately):
  figures/fig1_memory_scaling.{png,pdf}
  figures/fig2_throughput_scaling.{png,pdf}
  figures/fig3_memory_reduction.{png,pdf}

Method: SVD per-head (ra48/rf256/rw208), bs=32, fp32, avg. 8 GLUE tasks

Data sources:
  Naive(einsum): dde6df0 / 66aceb9  — full eval run, rank=ra48_rf256_rw208
  Naive(SDPA)  : 13b6d39            — perf-only run, sdpa mode
  FlashSVD     : dde6df0 / 66aceb9  — backend=flashsvd
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'figs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Data (exact values from benchmark logs)
# ─────────────────────────────────────────────────────────────────
SEQ = np.array([128, 256, 512])

MEM = {
    "einsum": np.array([559.0,  942.1,  2003.9]),
    "sdpa":   np.array([840.0,  1078.0, 1566.0]),
    "flash":  np.array([377.7,  484.8,   708.1]),
}
THR = {
    "einsum": np.array([1325., 530., 195.]),
    "sdpa":   np.array([1487., 756., 352.]),
    "flash":  np.array([1460., 725., 336.]),
}
MEM_RED = (MEM["einsum"] - MEM["flash"]) / MEM["einsum"] * 100  # 32.4 / 48.5 / 64.7

# ─────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────
COLORS = {
    "einsum": "#D32F2F",   # red    — naive baseline
    "sdpa":   "#F57C00",   # orange — SDPA ablation
    "flash":  "#1565C0",   # blue   — FlashSVD (protagonist)
}
LABELS = {
    "einsum": "Naive (einsum)",
    "sdpa":   "Naive (SDPA)",
    "flash":  "FlashSVD (Triton)",
}
DASHES = {
    "einsum": (4, 2),   # dashed  — baseline
    "sdpa":   (2, 2),   # dotted  — ablation mid-point
    "flash":  (),       # solid   — proposed
}
MARKERS = {"einsum": "o", "sdpa": "s", "flash": "^"}
LW, MS = 2.0, 8

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

# ─────────────────────────────────────────────────────────────────
# Shared x-axis helper
# ─────────────────────────────────────────────────────────────────
SUBTITLE = "SVD per-head (ra48/rf256/rw208)  ·  batch=32  ·  fp32  ·  averaged over 8 GLUE tasks"

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
# Figure 1 – Peak Memory  (all three backends; SDPA cross-over noted)
# SDPA data confirmed correct: peak_mem_infer_mb == peak_mem_e2e_mb.
# SDPA > einsum at short seq because fp32 SDPA uses Memory-Efficient
# Attention (not FA2, which needs fp16) with O(M) tile overhead that
# dominates over the O(M²) attention matrix savings at seq ≤ 256.
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.2))
for key in ("einsum", "sdpa", "flash"):
    ax.plot(SEQ, MEM[key],
            color=COLORS[key], marker=MARKERS[key],
            lw=LW, ms=MS,
            dashes=DASHES[key] if DASHES[key] else [],
            label=LABELS[key],
            zorder=4 if key == "flash" else 3)

ax.fill_between(SEQ, MEM["flash"], MEM["einsum"],
                color=COLORS["flash"], alpha=0.07)

# Annotate FlashSVD reduction vs einsum
for sl, me, mf in zip(SEQ, MEM["einsum"], MEM["flash"]):
    pct = (me - mf) / me * 100
    mid = (me + mf) / 2
    ax.text(sl + 14, mid, f"−{pct:.1f}%",
            color=COLORS["flash"], fontsize=9,
            va="center", ha="left", fontstyle="italic")

# Annotate SDPA cross-over with a subtle marker
ax.annotate(
    "SDPA > einsum\n(fp32 MEA overhead)",
    xy=(128, MEM["sdpa"][0]), xytext=(148, 1420),
    fontsize=7.5, color=COLORS["sdpa"],
    arrowprops=dict(arrowstyle="->", color=COLORS["sdpa"], lw=0.8),
)

ax.set_ylim(0, 2500)
ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
ax.set_ylabel("Peak GPU Memory (MB)", fontsize=11)
ax.set_title("Peak Memory vs. Sequence Length", fontweight="bold")
_setup_xax(ax)
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(axis="y", alpha=0.25, lw=0.8)
fig.text(0.5, -0.04, SUBTITLE, ha="center", fontsize=8.5, color="#555555")
fig.tight_layout()
_save(fig, "seqlen_memory")

# ─────────────────────────────────────────────────────────────────
# Figure 2 – Throughput
# naive lines: dashed; flash: solid + thicker (protagonist)
# annotate speedup only at seq=512 to avoid clutter
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.2))
for key in ("einsum", "sdpa", "flash"):
    lw_k = LW + 0.4 if key == "flash" else LW
    ax.plot(SEQ, THR[key],
            color=COLORS[key], marker=MARKERS[key],
            lw=lw_k, ms=MS,
            dashes=DASHES[key] if DASHES[key] else [],
            label=LABELS[key],
            zorder=4 if key == "flash" else 3)

ax.fill_between(SEQ, THR["einsum"], THR["flash"],
                color=COLORS["flash"], alpha=0.07)

# Annotate speedup only at seq=512
sl512, te512, tf512 = SEQ[-1], THR["einsum"][-1], THR["flash"][-1]
spd512 = tf512 / te512
ax.annotate(
    f"×{spd512:.2f} vs. Naive",
    xy=(sl512, tf512), xytext=(sl512 - 90, tf512 + 100),
    color=COLORS["flash"], fontsize=9.5, fontstyle="italic",
    arrowprops=dict(arrowstyle="->", color=COLORS["flash"], lw=1.0),
)

ax.set_ylim(0, 1700)
ax.yaxis.set_major_locator(ticker.MultipleLocator(300))
ax.set_ylabel("Throughput (samples / sec)", fontsize=11)
ax.set_title("Throughput vs. Sequence Length", fontweight="bold")
_setup_xax(ax)
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(axis="y", alpha=0.25, lw=0.8)
fig.text(0.5, -0.04, SUBTITLE, ha="center", fontsize=8.5, color="#555555")
fig.tight_layout()
_save(fig, "seqlen_throughput")

# ─────────────────────────────────────────────────────────────────
# Figure 3 – Memory Reduction (%) — single rising line
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.plot(SEQ, MEM_RED,
        color=COLORS["flash"], marker=MARKERS["flash"],
        lw=2.4, ms=MS, zorder=4, label="FlashSVD vs Naive(einsum)")
ax.fill_between(SEQ, 0, MEM_RED, color=COLORS["flash"], alpha=0.10)

for sl, r in zip(SEQ, MEM_RED):
    ax.text(sl, r + 1.5, f"{r:.1f}%",
            color=COLORS["flash"], fontsize=11,
            va="bottom", ha="center", fontweight="bold")

ax.set_ylim(0, 80)
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax.set_ylabel("Memory Reduction (%) relative to Naive (einsum)", fontsize=11)
ax.set_title("FlashSVD Memory Reduction vs. Sequence Length", fontweight="bold")
_setup_xax(ax)
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(axis="y", alpha=0.25, lw=0.8)
fig.text(0.5, -0.04, SUBTITLE, ha="center", fontsize=8.5, color="#555555")
fig.tight_layout()
_save(fig, "seqlen_reduction")

# ─────────────────────────────────────────────────────────────────
# Print final table
# ─────────────────────────────────────────────────────────────────
SEP = "=" * 70
print()
print(SEP)
print("Memory (MB)")
print(f"  {'seq':>4}   {'Naive(einsum)':>14}   {'Naive(SDPA)':>12}   {'FlashSVD':>10}   {'Reduction':>10}")
print("-" * 70)
for i, sl in enumerate(SEQ):
    print(f"  {sl:>4}   {MEM['einsum'][i]:>14.1f}   {MEM['sdpa'][i]:>12.1f}   "
          f"{MEM['flash'][i]:>10.1f}   {MEM_RED[i]:>9.1f}%")
print()
print("Throughput (sps)")
print(f"  {'seq':>4}   {'Naive(einsum)':>14}   {'Naive(SDPA)':>12}   {'FlashSVD':>10}   {'Speedup':>8}")
print("-" * 70)
for i, sl in enumerate(SEQ):
    print(f"  {sl:>4}   {THR['einsum'][i]:>14.0f}   {THR['sdpa'][i]:>12.0f}   "
          f"{THR['flash'][i]:>10.0f}   ×{THR['flash'][i]/THR['einsum'][i]:>6.2f}")
print(SEP)
