#!/usr/bin/env python3
"""
Seq-len and batch-size scaling figures from expC CSVs.

Output files:
  seqlen_memory.{png,pdf}
  seqlen_throughput.{png,pdf}
  seqlen_reduction.{png,pdf}
  batch_memory.{png,pdf}
  batch_throughput.{png,pdf}

Data: expC_seqlen.csv / expC_batch.csv
  bf16 · synthetic input (0% padding) · avg over methods and tasks
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

_REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIG_DIR = os.path.join(_REPO, 'experiments', 'figs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SEQLEN_CSV = os.path.join(_REPO, 'experiments', 'results', 'expC_seqlen.csv')
BATCH_CSV  = os.path.join(_REPO, 'experiments', 'results', 'expC_batch.csv')

# ── backend style ──────────────────────────────────────────────────────────────
BACKEND_ORDER  = ["naive", "sdpa", "flashsvd", "flashsvd15"]
COLORS  = {"naive": "#9e9e9e", "sdpa": "#bdbdbd",
           "flashsvd": "#42a5f5", "flashsvd15": "#ef5350"}
LABELS  = {"naive": "Naive (einsum)", "sdpa": "Naive (SDPA)",
           "flashsvd": "FlashSVD v1", "flashsvd15": "FlashSVD v1.5"}
DASHES  = {"naive": (4, 2), "sdpa": (2, 2), "flashsvd": (), "flashsvd15": ()}
MARKERS = {"naive": "o", "sdpa": "s", "flashsvd": "^", "flashsvd15": "D"}
LW, MS  = 2.0, 8

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


# ── data loading ───────────────────────────────────────────────────────────────
def _load(path, x_col):
    """
    Read CSV, filter bf16 + synthetic, average peak_mem_mb and throughput_sps
    over all methods and tasks, grouped by (backend, x_col).
    Returns dict: backend -> (x_array, mem_array, thr_array).
    """
    mem_acc = defaultdict(lambda: defaultdict(list))
    thr_acc = defaultdict(lambda: defaultdict(list))
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r.get('dtype', '').strip() != 'bf16':
                continue
            if r.get('input_mode', '').strip() != 'synthetic':
                continue
            b = r.get('backend', '').strip()
            try:
                x   = int(float(r[x_col]))
                mem = float(r['peak_mem_mb'])
                thr = float(r['throughput_sps'])
            except (KeyError, ValueError):
                continue
            mem_acc[b][x].append(mem)
            thr_acc[b][x].append(thr)

    result = {}
    for b in BACKEND_ORDER:
        if b not in mem_acc:
            continue
        xs  = sorted(mem_acc[b])
        mem = np.array([np.mean(mem_acc[b][x]) for x in xs])
        thr = np.array([np.mean(thr_acc[b][x]) for x in xs])
        result[b] = (np.array(xs), mem, thr)
    return result


def _save(fig, stem):
    p = os.path.join(FIG_DIR, f"{stem}.png")
    fig.savefig(p, dpi=180, bbox_inches='tight')
    print(f"Saved: {p}")
    plt.close(fig)


# ── seqlen figures ─────────────────────────────────────────────────────────────
SUBTITLE_SEQ = ("SVD per-head (ra48/rf256/rw208)  ·  batch=32  ·  bf16  ·  "
                "synthetic input  ·  avg over methods & tasks")

data_seq = _load(SEQLEN_CSV, 'seq_len')

# Reference backend for reductions: naive
_ref_b = "naive"
_ref_xs, _ref_mem, _ref_thr = data_seq[_ref_b]

# Figure 1 — Memory vs seq_len
fig, ax = plt.subplots(figsize=(5.5, 4.2))
for b in BACKEND_ORDER:
    if b not in data_seq:
        continue
    xs, mem, _ = data_seq[b]
    lw_b = LW + 0.4 if 'flash' in b else LW
    ax.plot(xs, mem, color=COLORS[b], marker=MARKERS[b],
            lw=lw_b, ms=MS,
            dashes=DASHES[b] if DASHES[b] else [],
            label=LABELS[b], zorder=4 if 'flash' in b else 3)

# Annotate FlashSVD v1.5 reduction vs naive — only at last (max) seq point
if "flashsvd15" in data_seq and _ref_b in data_seq:
    fxs, fmem, _ = data_seq["flashsvd15"]
    x_last, mf_last = fxs[-1], fmem[-1]
    idx_n = np.where(_ref_xs == x_last)[0]
    if len(idx_n):
        pct = (_ref_mem[idx_n[0]] - mf_last) / _ref_mem[idx_n[0]] * 100
        ax.text(x_last + 14, mf_last, f"−{pct:.0f}%\nvs Naive",
                color=COLORS["flashsvd15"], fontsize=9,
                va='center', ha='left', fontstyle='italic', fontweight='bold')

ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
ax.set_ylim(bottom=150)
ax.set_ylabel("Peak GPU Memory (MB)", fontsize=11)
ax.set_title("Peak Memory vs. Sequence Length", fontweight='bold')
ax.set_xticks(_ref_xs); ax.set_xticklabels([str(x) for x in _ref_xs])
ax.set_xlim(_ref_xs[0] - 32, _ref_xs[-1] + 80)
ax.set_xlabel("Sequence Length", fontsize=11)
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.25, lw=0.8)
fig.text(0.5, -0.04, SUBTITLE_SEQ, ha='center', fontsize=8.5, color='#555555')
fig.tight_layout()
_save(fig, "fig07_seqlen_A")

# Figure 2 — Throughput vs seq_len
fig, ax = plt.subplots(figsize=(5.5, 4.2))
for b in BACKEND_ORDER:
    if b not in data_seq:
        continue
    xs, _, thr = data_seq[b]
    lw_b = LW + 0.4 if 'flash' in b else LW
    ax.plot(xs, thr, color=COLORS[b], marker=MARKERS[b],
            lw=lw_b, ms=MS,
            dashes=DASHES[b] if DASHES[b] else [],
            label=LABELS[b], zorder=4 if 'flash' in b else 3)

# Annotate flashsvd15 speedup vs naive — only at last seq point
if "flashsvd15" in data_seq and _ref_b in data_seq:
    fxs, _, fthr = data_seq["flashsvd15"]
    x_last = fxs[-1];  tf_last = fthr[-1]
    idx_n = np.where(_ref_xs == x_last)[0]
    if len(idx_n):
        spd = tf_last / _ref_thr[idx_n[0]]
        ax.annotate(f"×{spd:.2f}",
                    xy=(x_last, tf_last), xytext=(x_last - 40, tf_last + 200),
                    color=COLORS["flashsvd15"], fontsize=9, fontstyle='italic',
                    ha='center',
                    arrowprops=dict(arrowstyle="->", color=COLORS["flashsvd15"],
                                    lw=0.8))

# Annotate SDPA path switch at seq=256 (short label)
if "sdpa" in data_seq:
    _sx, _, _sthr = data_seq["sdpa"]
    _idx256 = np.where(_sx == 256)[0]
    _idx128 = np.where(_sx == 128)[0]
    if len(_idx256) and len(_idx128):
        _t256 = _sthr[_idx256[0]]; _t128 = _sthr[_idx128[0]]
        if _t256 > _t128:
            ax.annotate("path switch",
                        xy=(256, _t256), xytext=(196, _t256),
                        color=COLORS["sdpa"], fontsize=8.5, fontstyle="italic",
                        ha="center",
                        arrowprops=dict(arrowstyle="->", color=COLORS["sdpa"], lw=1.0))

ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
ax.set_ylabel("Throughput (samples / sec)", fontsize=11)
ax.set_title("Throughput vs. Sequence Length", fontweight='bold')
ax.set_xticks(_ref_xs); ax.set_xticklabels([str(x) for x in _ref_xs])
ax.set_xlim(_ref_xs[0] - 32, _ref_xs[-1] + 48)
ax.set_xlabel("Sequence Length", fontsize=11)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.25, lw=0.8)
fig.text(0.5, -0.04, SUBTITLE_SEQ, ha='center', fontsize=8.5, color='#555555')
fig.tight_layout()
_save(fig, "fig07_seqlen_B")

# Figure 3 — Memory Reduction (%) vs seq_len
fig, ax = plt.subplots(figsize=(5.5, 4.2))
_placed = {}   # (x, r_rounded) -> count, for label deconfliction
_red_data = []
for b in [b for b in BACKEND_ORDER if b != _ref_b]:
    if b not in data_seq:
        continue
    fxs, fmem, _ = data_seq[b]
    mask = np.isin(fxs, _ref_xs)
    xs_a = fxs[mask]
    red  = (_ref_mem[np.isin(_ref_xs, xs_a)] - fmem[mask]) / _ref_mem[np.isin(_ref_xs, xs_a)] * 100
    lw_b = 2.4 if 'flash' in b else LW
    ax.plot(xs_a, red, color=COLORS[b], marker=MARKERS[b],
            lw=lw_b, ms=MS,
            dashes=DASHES[b] if DASHES[b] else [],
            label=f"{LABELS[b]} vs Naive", zorder=4 if 'flash' in b else 3)
    ax.fill_between(xs_a, 0, red, color=COLORS[b], alpha=0.06)
    _red_data.append((b, xs_a, red))

# Place labels: skip if another backend already placed a label within 3% at same x
_label_placed = {}   # x -> list of r values already labeled
for b, xs_a, red in _red_data:
    for x, r in zip(xs_a, red):
        prev = _label_placed.get(x, [])
        if any(abs(r - p) < 3 for p in prev):
            continue   # too close to an existing label — skip
        _label_placed.setdefault(x, []).append(r)
        ax.text(x, r + 1.5, f"{r:.0f}%",
                color=COLORS[b], fontsize=9,
                va='bottom', ha='center', fontweight='bold')

ax.set_ylim(0, 80)
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax.set_ylabel("Memory Reduction (%) vs. Naive (einsum)", fontsize=11)
ax.set_title("Memory Reduction vs. Sequence Length", fontweight='bold')
ax.set_xticks(_ref_xs); ax.set_xticklabels([str(x) for x in _ref_xs])
ax.set_xlim(_ref_xs[0] - 32, _ref_xs[-1] + 48)
ax.set_xlabel("Sequence Length", fontsize=11)
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.25, lw=0.8)
fig.text(0.5, -0.04, SUBTITLE_SEQ, ha='center', fontsize=8.5, color='#555555')
fig.tight_layout()
_save(fig, "fig07_seqlen_C")


# ── batch figures ──────────────────────────────────────────────────────────────
SUBTITLE_BATCH = ("SVD per-head (ra48/rf256/rw208)  ·  seq_len=512  ·  bf16  ·  "
                  "synthetic input  ·  avg over methods & tasks")

data_batch = _load(BATCH_CSV, 'batch_size')
_ref_bxs, _ref_bmem, _ref_bthr = data_batch[_ref_b]

# Figure 4 — Memory vs batch_size
fig, ax = plt.subplots(figsize=(5.5, 4.2))
for b in BACKEND_ORDER:
    if b not in data_batch:
        continue
    xs, mem, _ = data_batch[b]
    lw_b = LW + 0.4 if 'flash' in b else LW
    ax.plot(xs, mem, color=COLORS[b], marker=MARKERS[b],
            lw=lw_b, ms=MS,
            dashes=DASHES[b] if DASHES[b] else [],
            label=LABELS[b], zorder=4 if 'flash' in b else 3)

ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
ax.set_ylabel("Peak GPU Memory (MB)", fontsize=11)
ax.set_title("Peak Memory vs. Batch Size", fontweight='bold')
ax.set_xticks(_ref_bxs); ax.set_xticklabels([str(x) for x in _ref_bxs])
ax.set_xlim(_ref_bxs[0] - 4, _ref_bxs[-1] + 8)
ax.set_xlabel("Batch Size", fontsize=11)
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.25, lw=0.8)
fig.text(0.5, -0.04, SUBTITLE_BATCH, ha='center', fontsize=8.5, color='#555555')
fig.tight_layout()
_save(fig, "fig08_batch_A")

# Figure 5 — Throughput vs batch_size
fig, ax = plt.subplots(figsize=(5.5, 4.2))
for b in BACKEND_ORDER:
    if b not in data_batch:
        continue
    xs, _, thr = data_batch[b]
    lw_b = LW + 0.4 if 'flash' in b else LW
    ax.plot(xs, thr, color=COLORS[b], marker=MARKERS[b],
            lw=lw_b, ms=MS,
            dashes=DASHES[b] if DASHES[b] else [],
            label=LABELS[b], zorder=4 if 'flash' in b else 3)

ax.yaxis.set_major_locator(ticker.MultipleLocator(400))
ax.set_ylabel("Throughput (samples / sec)", fontsize=11)
ax.set_title("Throughput vs. Batch Size", fontweight='bold')
ax.set_xticks(_ref_bxs); ax.set_xticklabels([str(x) for x in _ref_bxs])
ax.set_xlim(_ref_bxs[0] - 4, _ref_bxs[-1] + 8)
ax.set_xlabel("Batch Size", fontsize=11)
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, framealpha=0.9)
ax.grid(axis='y', alpha=0.25, lw=0.8)
fig.text(0.5, -0.04, SUBTITLE_BATCH, ha='center', fontsize=8.5, color='#555555')
fig.tight_layout()
_save(fig, "fig08_batch_B")


# ── summary table ──────────────────────────────────────────────────────────────
SEP = "=" * 72
print()
print(SEP)
print("Seq-len scaling  (bf16, synthetic, avg methods+tasks)")
hdr = f"  {'seq':>4}  " + "  ".join(f"{LABELS[b]:>18}" for b in BACKEND_ORDER if b in data_seq)
print(hdr + "  (mem MB)")
print("-" * 72)
for i, x in enumerate(_ref_xs):
    row = f"  {x:>4}  "
    for b in BACKEND_ORDER:
        if b not in data_seq: continue
        row += f"  {data_seq[b][1][i]:>18.1f}"
    print(row)
print()
print(hdr.replace("(mem MB)", "(thr sps)"))
print("-" * 72)
for i, x in enumerate(_ref_xs):
    row = f"  {x:>4}  "
    for b in BACKEND_ORDER:
        if b not in data_seq: continue
        row += f"  {data_seq[b][2][i]:>18.0f}"
    print(row)
print(SEP)
