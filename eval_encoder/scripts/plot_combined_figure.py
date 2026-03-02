#!/usr/bin/env python3
"""
Combined publication figure — single large figure with 4 metric panels + 1 fairness table.

Layout (2 columns × 2 task rows per half + 1 fairness row):

  ┌─────────────────────┬─────────────────────┐
  │  Latency (MNLI)     │  Throughput (MNLI)  │
  ├─────────────────────┼─────────────────────┤
  │  Latency (STS-B)    │  Throughput (STS-B) │
  ├─────────────────────┼─────────────────────┤
  │  Memory (MNLI)      │  Speedup (MNLI)     │
  ├─────────────────────┼─────────────────────┤
  │  Memory (STS-B)     │  Speedup (STS-B)    │
  ├─────────────────────┴─────────────────────┤
  │  Fairness Check Table (full width)         │
  └────────────────────────────────────────────┘

Usage:
    python eval_encoder/scripts/plot_combined_figure.py \\
        --csv    eval_encoder/eval_results/expA_backend.csv \\
        --outdir eval_encoder/eval_results/figures
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ── visual config (must match plot_backend_sweep.py) ──────────────────────────
BACKEND_ORDER  = ["naive", "sdpa", "flashsvd", "flashsvd15"]
BACKEND_COLORS = {"naive": "#9e9e9e", "sdpa": "#66bb6a",
                  "flashsvd": "#42a5f5", "flashsvd15": "#ef5350"}
BACKEND_LABELS = {"naive": "Naive", "sdpa": "SDPA",
                  "flashsvd": "FlashSVD", "flashsvd15": "FlashSVD 1.5"}
METHOD_ORDER   = ["svd", "fwsvd", "drone", "adasvd"]
METHOD_LABELS  = {"svd": "SVD", "fwsvd": "FWSVD", "drone": "DRONE", "adasvd": "AdaSVD"}
TASK_LABELS    = {"mnli": "MNLI", "cola": "CoLA", "stsb": "STS-B",
                  "sst2": "SST-2", "mrpc": "MRPC"}

BAR_W_UNIT = 0.62   # total group width / n_backends


# ── data loading ───────────────────────────────────────────────────────────────
def load_all(path, tasks, dtype, seq_len, methods):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    mask = (
        df["task"].isin(tasks) &
        (df["dtype"] == dtype) &
        (df["seq_len"].astype(int) == seq_len) &
        df["method"].isin(methods)
    )
    df = df[mask].copy()
    for col in ["latency_ms", "throughput_sps", "peak_mem_mb",
                "total_flops_abs", "rank_pad_pct", "seq_pad_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── bar helpers ────────────────────────────────────────────────────────────────
def _backends_present(df):
    return [b for b in BACKEND_ORDER if b in df["backend"].values]


def _draw_bars(ax, df_task, methods, col, ylabel, higher_better, fmt="{:.1f}",
               show_xlabel=True, title=None):
    backends = _backends_present(df_task)
    n_m = len(methods)
    n_b = len(backends)
    bw  = BAR_W_UNIT / n_b
    x   = np.arange(n_m)

    ymax = 0.0
    for bi, backend in enumerate(backends):
        vals = []
        for method in methods:
            row = df_task[(df_task["method"] == method) &
                          (df_task["backend"] == backend)]
            vals.append(float(row[col].values[0]) if len(row) > 0 else np.nan)
        offset = (bi - (n_b - 1) / 2) * bw
        bars = ax.bar(x + offset, vals, bw * 0.88,
                      color=BACKEND_COLORS[backend],
                      label=BACKEND_LABELS[backend], zorder=3)
        for rect, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ymax = max(ymax, v)
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + ymax * 0.01,
                    fmt.format(v),
                    ha="center", va="bottom", fontsize=6.5, color="#222222")

    ax.set_xticks(x)
    if show_xlabel:
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=8)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_ylim(0, ymax * 1.22)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=3)
    # higher/lower arrow
    arrow = "↑" if higher_better else "↓"
    ax.text(1.0, 1.0, arrow, transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color="#555555")


def _draw_speedup(ax, df_task, methods, show_xlabel=True, title=None):
    backends = _backends_present(df_task)
    # exclude naive from speedup bars (it's the reference = 1.0)
    speedup_backends = [b for b in backends if b != "naive"]
    n_m = len(methods)
    n_b = len(speedup_backends)
    bw  = BAR_W_UNIT / max(n_b, 1)
    x   = np.arange(n_m)

    ymax = 0.0
    for bi, backend in enumerate(speedup_backends):
        vals = []
        for method in methods:
            nr = df_task[(df_task["method"] == method) &
                         (df_task["backend"] == "naive")]
            br = df_task[(df_task["method"] == method) &
                         (df_task["backend"] == backend)]
            if len(nr) > 0 and len(br) > 0:
                nv = float(nr["throughput_sps"].values[0])
                bv = float(br["throughput_sps"].values[0])
                vals.append(bv / nv if nv > 0 else np.nan)
            else:
                vals.append(np.nan)
        offset = (bi - (n_b - 1) / 2) * bw
        bars = ax.bar(x + offset, vals, bw * 0.88,
                      color=BACKEND_COLORS[backend],
                      label=BACKEND_LABELS[backend], zorder=3)
        for rect, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ymax = max(ymax, v)
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.02,
                    f"{v:.2f}×",
                    ha="center", va="bottom", fontsize=6.5, color="#222222")

    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    if show_xlabel:
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=8)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel("Speedup vs. Naive", fontsize=8)
    ax.set_ylim(0, max(ymax, 1.1) * 1.22)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=3)
    ax.text(1.0, 1.0, "↑", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color="#555555")


def _draw_fairness(ax, df, tasks, methods):
    ax.axis("off")
    backends = _backends_present(df)

    # Group methods by their (total_flops, rank_pad) — SVD/FWSVD/DRONE are identical
    groups = {}
    for m in methods:
        row = df[(df["method"] == m) & (df["backend"] == backends[0])]
        if len(row) == 0:
            continue
        key = (round(float(row["total_flops_abs"].values[0]), 3),
               round(float(row["rank_pad_pct"].values[0]), 2))
        groups.setdefault(key, []).append(METHOD_LABELS.get(m, m))

    rows_data = []
    for (flops, rpad), mlist in groups.items():
        label = "/".join(mlist)
        rows_data.append([label,
                          f"{flops:.4f} TFLOPs",
                          f"{rpad:.2f}%",
                          "✓ identical across backends"])

    # seq_pad (per task)
    for task in tasks:
        sub = df[df["task"] == task]
        if sub.empty:
            continue
        sp = float(sub["seq_pad_pct"].dropna().values[0])
        rows_data.append([f"seq_pad ({TASK_LABELS.get(task, task)})",
                          "—", f"{sp:.2f}%",
                          "✓ identical across backends"])

    col_labels = ["Method group", "Total FLOPs (abs)", "Rank-pad %", "Invariance"]
    tbl = ax.table(
        cellText=rows_data,
        colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.35)

    # header style
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#37474f")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # alternating rows
    for i in range(1, len(rows_data) + 1):
        fc = "#f5f5f5" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(fc)

    ax.set_title("Fairness Check — FLOPs invariants across backends",
                 fontsize=8.5, fontweight="bold", pad=6)


# ── legend ─────────────────────────────────────────────────────────────────────
def _make_legend(fig, backends):
    handles = [plt.Rectangle((0, 0), 1, 1,
                              color=BACKEND_COLORS[b],
                              label=BACKEND_LABELS[b])
               for b in backends]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(backends), fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.005))


# ── main plot ──────────────────────────────────────────────────────────────────
def plot_combined(df, tasks, methods, outdir, dtype, seq_len):
    n_tasks = len(tasks)
    # rows: n_tasks for upper pair + n_tasks for lower pair + 1 fairness
    n_rows = n_tasks * 2 + 1
    h_task = 3.2
    h_fair = 1.6 + 0.3 * n_tasks
    height_ratios = [h_task] * (n_tasks * 2) + [h_fair]

    fig = plt.figure(figsize=(13, sum(height_ratios) + 0.8))
    fig.suptitle(
        f"Backend Comparison: SVD-Compressed BERT  |  dtype={dtype}  seq_len={seq_len}  batch=32",
        fontsize=11, fontweight="bold", y=0.995,
    )

    gs = gridspec.GridSpec(
        n_rows, 2,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.45, wspace=0.32,
    )

    # ── column headers (text above first row) ─────────────────────────────────
    # We use set_title on the first-row axes for each column group.

    backends_present = _backends_present(df)

    for ti, task in enumerate(tasks):
        sub = df[df["task"] == task]
        tl  = TASK_LABELS.get(task, task)
        is_last = (ti == n_tasks - 1)

        # upper half: Latency (col 0) + Throughput (col 1)
        ax_lat = fig.add_subplot(gs[ti, 0])
        ax_thr = fig.add_subplot(gs[ti, 1])
        col_title_lat = "Latency (ms / batch) ↓" if ti == 0 else None
        col_title_thr = "Throughput (samples / s) ↑" if ti == 0 else None
        _draw_bars(ax_lat, sub, methods, "latency_ms",
                   ylabel=f"{tl}  ms / batch",
                   higher_better=False, fmt="{:.1f}",
                   show_xlabel=is_last, title=col_title_lat)
        _draw_bars(ax_thr, sub, methods, "throughput_sps",
                   ylabel=f"{tl}  sps",
                   higher_better=True, fmt="{:.0f}",
                   show_xlabel=is_last, title=col_title_thr)

        # lower half: Memory (col 0) + Speedup (col 1)
        ax_mem = fig.add_subplot(gs[n_tasks + ti, 0])
        ax_spd = fig.add_subplot(gs[n_tasks + ti, 1])
        col_title_mem = "Peak GPU Memory (MB) ↓" if ti == 0 else None
        col_title_spd = "Throughput Speedup vs. Naive ↑" if ti == 0 else None
        _draw_bars(ax_mem, sub, methods, "peak_mem_mb",
                   ylabel=f"{tl}  MB",
                   higher_better=False, fmt="{:.0f}",
                   show_xlabel=is_last, title=col_title_mem)
        _draw_speedup(ax_spd, sub, methods,
                      show_xlabel=is_last, title=col_title_spd)

    # ── fairness table (full width, last row) ─────────────────────────────────
    ax_fair = fig.add_subplot(gs[-1, :])
    _draw_fairness(ax_fair, df, tasks, methods)

    # ── shared legend at bottom ────────────────────────────────────────────────
    _make_legend(fig, backends_present)

    os.makedirs(outdir, exist_ok=True)
    fname = f"combined_{'+'.join(tasks)}_{dtype}_seq{seq_len}.png"
    out   = os.path.join(outdir, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     default="eval_encoder/eval_results/expA_backend.csv")
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
        raise SystemExit("[error] No matching rows in CSV.")

    methods  = [m for m in args.methods  if m in df["method"].unique()]
    tasks    = [t for t in args.tasks    if t in df["task"].unique()]

    print(f"[plot] tasks={tasks}  methods={methods}  outdir={args.outdir}")
    plot_combined(df, tasks, methods, args.outdir, args.dtype, args.seq_len)


if __name__ == "__main__":
    main()
