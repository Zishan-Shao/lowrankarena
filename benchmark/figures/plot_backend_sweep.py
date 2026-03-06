#!/usr/bin/env python3
"""
Plot backend comparison — one figure per (metric × task).

Backends plotted: SDPA, FlashSVD 1.0, FlashSVD 1.5  (Naive used only as speedup denominator)
Highlights:       AdaSVD + FlashSVD 1.5  → bold border + ★ label

Usage:
    python benchmark/figures/plot_backend_sweep.py \
        --csv     experiments/results/expB.csv \
        --tasks   mnli stsb \
        --methods svd fwsvd drone adasvd \
        --dtype   bf16 --seq_len 512 \
        --outdir  experiments/figs/figures
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory


# ── visual config ──────────────────────────────────────────────────────────────
# Naive excluded from plots (used only as speedup baseline)
BACKEND_ORDER  = ["sdpa", "flashsvd", "flashsvd15"]
BACKEND_COLORS = {
    "sdpa":       "#bdbdbd",   # light gray
    "flashsvd":   "#42a5f5",   # blue
    "flashsvd15": "#ef5350",   # red — proposed best backend
}
BACKEND_LABELS = {
    "sdpa":       "SDPA",
    "flashsvd":   "FlashSVD 1.0",
    "flashsvd15": "FlashSVD 1.5",
}
METHOD_LABELS = {"svd": "SVD", "fwsvd": "FWSVD", "drone": "DRONE", "adasvd": "AdaSVD"}
TASK_LABELS   = {"mnli": "MNLI", "cola": "CoLA", "stsb": "STS-B",
                 "sst2": "SST-2", "mrpc": "MRPC", "qqp": "QQP",
                 "qnli": "QNLI", "rte": "RTE"}
TASK_SUFFIX   = "ABCDEFGHIJ"

METRIC_FIG_NUM   = {"latency": "09", "throughput": "10", "speedup": "11", "memory": "12"}
HIGHLIGHT_BACKEND = "flashsvd15"
HIGHLIGHT_METHOD  = "adasvd"


def load_all(path, tasks, dtype, seq_len, methods):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    # Keep all backends in df (naive needed for speedup denominator)
    mask = (
        df["task"].isin(tasks) &
        (df["dtype"] == dtype) &
        (df["seq_len"].astype(int) == seq_len) &
        df["method"].isin(methods)
    )
    df = df[mask].copy()
    for col in ["latency_ms", "throughput_sps", "peak_mem_mb",
                "rank_pad_pct", "seq_pad_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "input_mode" in df.columns:
        syn = df[df["input_mode"] == "synthetic"]
        df  = syn if not syn.empty else df
    df = df.drop_duplicates(subset=["task", "method", "backend"])
    return df


# ── bar helpers ────────────────────────────────────────────────────────────────

def _bar_vals(df_task, methods, col):
    """Return {backend: [val_per_method]} for BACKEND_ORDER only."""
    vals = {}
    for b in BACKEND_ORDER:
        row = []
        for m in methods:
            r = df_task[(df_task["backend"] == b) & (df_task["method"] == m)]
            row.append(float(r[col].values[0]) if len(r) > 0 else 0.0)
        vals[b] = row
    return vals


def _draw_bars(ax, df_task, methods, col, ylabel, higher_better, fmt="{:.1f}"):
    bv  = _bar_vals(df_task, methods, col)
    n_m = len(methods)
    n_b = len(BACKEND_ORDER)
    bw  = 0.72 / n_b   # wider per-backend bar since fewer backends
    x   = np.arange(n_m)

    all_vals = [v for b in BACKEND_ORDER for v in bv[b] if v > 0]
    ymax = max(all_vals) if all_vals else 1.0

    for bi, backend in enumerate(BACKEND_ORDER):
        offset = (bi - (n_b - 1) / 2) * bw
        for mi, (m, v) in enumerate(zip(methods, bv[backend])):
            is_star = (backend == HIGHLIGHT_BACKEND and m == HIGHLIGHT_METHOD)
            ax.bar(x[mi] + offset, v, width=bw * 0.88,
                   color=BACKEND_COLORS[backend], zorder=3,
                   edgecolor="#111111" if is_star else "none",
                   linewidth=2.0 if is_star else 0,
                   label=BACKEND_LABELS[backend] if mi == 0 else "")
            if v > 0:
                txt = fmt.format(v) + (" ★" if is_star else "")
                ax.text(x[mi] + offset,
                        v + ymax * 0.018,
                        txt,
                        ha="center", va="bottom",
                        fontsize=8 if backend != HIGHLIGHT_BACKEND else 9,
                        color="#111111",
                        fontweight="bold" if is_star else "normal")

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, ymax * 1.28)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=11)
    ax.text(0.99, 0.97, "↑" if higher_better else "↓",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=13, color="#555555", fontweight="bold")


def _draw_speedup(ax, df_task, methods):
    n_m = len(methods)
    n_b = len(BACKEND_ORDER)
    bw  = 0.72 / n_b
    x   = np.arange(n_m)

    spd = {}
    all_spd = []
    for b in BACKEND_ORDER:
        row = []
        for m in methods:
            nr = df_task[(df_task["method"] == m) & (df_task["backend"] == "naive")]
            br = df_task[(df_task["method"] == m) & (df_task["backend"] == b)]
            if len(nr) > 0 and len(br) > 0:
                nv = float(nr["throughput_sps"].values[0])
                bv = float(br["throughput_sps"].values[0])
                s  = bv / nv if nv > 0 else 0.0
            else:
                s = 0.0
            row.append(s)
            if s > 0:
                all_spd.append(s)
        spd[b] = row

    ymax  = max(all_spd) if all_spd else 3.0
    y_min = 0.9   # tight y-range to emphasize differences

    for bi, backend in enumerate(BACKEND_ORDER):
        offset = (bi - (n_b - 1) / 2) * bw
        for mi, (m, v) in enumerate(zip(methods, spd[backend])):
            is_star = (backend == HIGHLIGHT_BACKEND and m == HIGHLIGHT_METHOD)
            ax.bar(x[mi] + offset, v, width=bw * 0.88,
                   color=BACKEND_COLORS[backend], zorder=3,
                   edgecolor="#111111" if is_star else "none",
                   linewidth=2.0 if is_star else 0,
                   label=BACKEND_LABELS[backend] if mi == 0 else "")
            if v > 0:
                txt = f"{v:.2f}×" + (" ★" if is_star else "")
                ax.text(x[mi] + offset,
                        v + (ymax - y_min) * 0.025,
                        txt,
                        ha="center", va="bottom",
                        fontsize=8 if backend != HIGHLIGHT_BACKEND else 9,
                        color="#111111",
                        fontweight="bold" if is_star else "normal")

    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0, zorder=2)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.02, 1.0, "Naive baseline (1.0×)",
            transform=trans, fontsize=8.5, color="#888888",
            va="center", ha="left", clip_on=False)
    ax.set_ylabel("Throughput Speedup vs Naive", fontsize=11)
    ax.set_ylim(y_min, ymax * 1.28)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=11)
    ax.text(0.99, 0.97, "↑", transform=ax.transAxes,
            ha="right", va="top", fontsize=13, color="#555555", fontweight="bold")
    return max(spd[HIGHLIGHT_BACKEND]) if spd.get(HIGHLIGHT_BACKEND) else 0.0


# ── per-(metric, task) figure ──────────────────────────────────────────────────

def _make_single(df, task, methods, metric, dtype, seq_len, suffix, outdir):
    sub        = df[df["task"] == task]
    task_label = TASK_LABELS.get(task, task)
    fig_num    = METRIC_FIG_NUM.get(metric, "XX")

    fig, ax = plt.subplots(figsize=(8, 4.8))

    if metric == "latency":
        title = f"Latency  —  {task_label}  |  dtype={dtype}  seq={seq_len}"
        _draw_bars(ax, sub, methods, "latency_ms", "ms / batch",
                   higher_better=False, fmt="{:.1f}")
    elif metric == "throughput":
        title = f"Throughput  —  {task_label}  |  dtype={dtype}  seq={seq_len}"
        _draw_bars(ax, sub, methods, "throughput_sps", "samples / s",
                   higher_better=True, fmt="{:.0f}")
    elif metric == "speedup":
        max_spd = _draw_speedup(ax, sub, methods)
        title = (f"FlashSVD 1.5 Achieves Up to {max_spd:.2f}× Speedup  —  "
                 f"{task_label}  |  dtype={dtype}  seq={seq_len}")
    elif metric == "memory":
        title = f"Peak GPU Memory  —  {task_label}  |  dtype={dtype}  seq={seq_len}"
        _draw_bars(ax, sub, methods, "peak_mem_mb", "MB",
                   higher_better=False, fmt="{:.0f}")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=BACKEND_COLORS[b],
                              label=BACKEND_LABELS[b]) for b in BACKEND_ORDER]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
              borderaxespad=0, fontsize=10, framealpha=0.9)

    fname = f"fig{fig_num}_{metric}_{suffix}.png"
    os.makedirs(outdir, exist_ok=True)
    path  = os.path.join(outdir, fname)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {path}")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     default="experiments/results/expB.csv")
    p.add_argument("--outdir",  default=None)
    p.add_argument("--tasks",   nargs="+", default=["mnli", "stsb"])
    p.add_argument("--methods", nargs="+", default=["svd", "fwsvd", "drone", "adasvd"])
    p.add_argument("--dtype",   default="bf16")
    p.add_argument("--seq_len", type=int, default=512)
    args = p.parse_args()

    if args.outdir is None:
        args.outdir = os.path.join(os.path.dirname(args.csv), "figures")

    df = load_all(args.csv, args.tasks, args.dtype, args.seq_len, args.methods)
    if df.empty:
        raise SystemExit("[error] No rows matched.")

    methods = [m for m in args.methods if m in df["method"].unique()]
    tasks   = [t for t in args.tasks   if t in df["task"].unique()]
    print(f"[plot] tasks={tasks}  methods={methods}  outdir={args.outdir}")

    for ti, task in enumerate(tasks):
        suffix = TASK_SUFFIX[ti] if ti < len(TASK_SUFFIX) else str(ti)
        for metric in ["latency", "throughput", "speedup", "memory"]:
            _make_single(df, task, methods, metric,
                         args.dtype, args.seq_len,
                         suffix, args.outdir)


if __name__ == "__main__":
    main()
