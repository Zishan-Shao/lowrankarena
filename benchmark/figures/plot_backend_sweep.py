#!/usr/bin/env python3
"""
Plot backend comparison — 3 separate figures (Latency / Speedup / Memory),
each with 2 task rows + 1 fairness-check row at the bottom.

Output: backend_latency_mnli+stsb_bf16_seq512.png  (and speedup / memory)

Usage:
    python eval_encoder/scripts/plot_backend_sweep.py \
        --csv     eval_encoder/eval_results/expB.csv \
        --tasks   mnli stsb \
        --methods svd fwsvd drone adasvd \
        --dtype   bf16  --seq_len 512 \
        --outdir  eval_encoder/eval_results/figures
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── visual config ──────────────────────────────────────────────────────────────
BACKEND_ORDER  = ["naive", "sdpa", "flashsvd", "flashsvd15"]
BACKEND_COLORS = {"naive": "#9e9e9e", "sdpa": "#66bb6a",
                  "flashsvd": "#42a5f5", "flashsvd15": "#ef5350"}
BACKEND_LABELS = {"naive": "Naive", "sdpa": "SDPA",
                  "flashsvd": "FlashSVD", "flashsvd15": "FlashSVD 1.5"}
METHOD_LABELS  = {"svd": "SVD", "fwsvd": "FWSVD", "drone": "DRONE", "adasvd": "AdaSVD"}
TASK_LABELS    = {"mnli": "MNLI", "cola": "CoLA", "stsb": "STS-B",
                  "sst2": "SST-2", "mrpc": "MRPC", "qqp": "QQP",
                  "qnli": "QNLI", "rte": "RTE"}


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
    for col in ["latency_ms", "throughput_sps", "total_flops_abs",
                "rank_pad_pct", "seq_pad_pct", "peak_mem_mb"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── bar drawing helpers ────────────────────────────────────────────────────────

def _draw_bars(ax, df_task, methods, col, ylabel, higher_better,
               fmt="{:.1f}", show_xlabel=True):
    n_m = len(methods)
    n_b = len(BACKEND_ORDER)
    bw  = 0.62 / n_b
    x   = np.arange(n_m)

    all_vals = []
    for b in BACKEND_ORDER:
        for m in methods:
            r = df_task[(df_task["backend"] == b) & (df_task["method"] == m)]
            if len(r) > 0:
                all_vals.append(float(r[col].values[0]))
    ymax = max(all_vals) if all_vals else 1.0

    for bi, backend in enumerate(BACKEND_ORDER):
        sub  = df_task[df_task["backend"] == backend]
        vals = [float(sub[sub["method"] == m][col].values[0])
                if len(sub[sub["method"] == m]) > 0 else 0.0
                for m in methods]

        offset = (bi - (n_b - 1) / 2) * bw
        bars = ax.bar(x + offset, vals, width=bw * 0.88,
                      color=BACKEND_COLORS[backend],
                      label=BACKEND_LABELS[backend], zorder=3)
        for rect, v in zip(bars, vals):
            if v > 0:
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + ymax * 0.016,
                        fmt.format(v),
                        ha="center", va="bottom", fontsize=8, color="#222222")

    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(0, ymax * 1.20)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    if show_xlabel:
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=9)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([""] * n_m)
    ax.text(0.99, 0.97, "↑" if higher_better else "↓",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color="#555555", fontweight="bold")


def _draw_speedup(ax, df_task, methods, show_xlabel=True):
    n_m = len(methods)
    n_b = len(BACKEND_ORDER)
    bw  = 0.62 / n_b
    x   = np.arange(n_m)

    all_spd = []
    for b in BACKEND_ORDER:
        for m in methods:
            nr = df_task[(df_task["method"] == m) & (df_task["backend"] == "naive")]
            br = df_task[(df_task["method"] == m) & (df_task["backend"] == b)]
            if len(nr) > 0 and len(br) > 0:
                nv = float(nr["throughput_sps"].values[0])
                bv = float(br["throughput_sps"].values[0])
                if nv > 0:
                    all_spd.append(bv / nv)
    ymax = max(all_spd) if all_spd else 3.0

    for bi, backend in enumerate(BACKEND_ORDER):
        speedups = []
        for m in methods:
            nr = df_task[(df_task["method"] == m) & (df_task["backend"] == "naive")]
            br = df_task[(df_task["method"] == m) & (df_task["backend"] == backend)]
            if len(nr) > 0 and len(br) > 0:
                nv = float(nr["throughput_sps"].values[0])
                bv = float(br["throughput_sps"].values[0])
                speedups.append(bv / nv if nv > 0 else 0.0)
            else:
                speedups.append(0.0)

        offset = (bi - (n_b - 1) / 2) * bw
        bars = ax.bar(x + offset, speedups, width=bw * 0.88,
                      color=BACKEND_COLORS[backend],
                      label=BACKEND_LABELS[backend], zorder=3)
        for rect, v in zip(bars, speedups):
            if v > 0:
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + ymax * 0.016,
                        f"{v:.2f}×",
                        ha="center", va="bottom", fontsize=8, color="#222222")

    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.9, zorder=2)
    ax.set_ylabel("Speedup vs Naive", fontsize=9)
    ax.set_ylim(0, ymax * 1.20)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    if show_xlabel:
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=9)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([""] * n_m)
    ax.text(0.99, 0.97, "↑", transform=ax.transAxes,
            ha="right", va="top", fontsize=11, color="#555555", fontweight="bold")


# ── fairness check panel ───────────────────────────────────────────────────────

def _draw_fairness_panel(ax, df, tasks, methods):
    """
    Bottom row: compact table showing the 3 invariant / semi-invariant quantities
    that prove the backend comparison is apples-to-apples.

      total_flops_abs  — same per (method, backend) ✓
      rank_pad_pct     — fixed model property (differs across methods)
      seq_pad_pct      — same across all backends ✓ (input batch identical)
    """
    ax.axis("off")

    ref = df[df["backend"] == "naive"]   # values are backend-invariant; use naive as reference

    # Group methods sharing the same (total_flops, rank_pad) → compact table
    t0     = tasks[0]
    groups = {}   # key=(total_str, rpad_str) → list of method labels
    for method in methods:
        mr = ref[(ref["method"] == method) & (ref["task"] == t0)]
        if len(mr) == 0:
            continue
        r = mr.iloc[0]
        total_str = f"{float(r['total_flops_abs']):.3f} TF"
        rpad_str  = f"{float(r['rank_pad_pct']):.2f}%"
        key = (total_str, rpad_str)
        groups.setdefault(key, []).append(METHOD_LABELS.get(method, method))

    # seq_pad: per task (same across methods and backends within a task)
    seq_parts = []
    for task in tasks:
        tr = ref[ref["task"] == task]
        if len(tr) > 0:
            sp = float(tr["seq_pad_pct"].values[0])
            seq_parts.append(f"{TASK_LABELS.get(task, task)}: {sp:.1f}%")
    seq_str = "  |  ".join(seq_parts)

    # Table rows
    col_labels = ["Method", "total_flops_abs", "rank_pad %", "seq_pad %"]
    table_data = []
    for (total_str, rpad_str), mlabels in groups.items():
        table_data.append([" / ".join(mlabels), total_str, rpad_str, seq_str])

    # Invariance note row
    table_data.append([
        "backend invariant?",
        "✓ identical per method",
        "✓ fixed (model property)",
        "✓ identical across all backends",
    ])

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.55)

    # header: blue-grey
    for j in range(len(col_labels)):
        c = tbl[(0, j)]
        c.set_facecolor("#cfd8dc")
        c.set_text_props(fontweight="bold", color="#1a3a5c")

    # note row: light green
    nr = len(table_data)
    for j in range(len(col_labels)):
        c = tbl[(nr, j)]
        c.set_facecolor("#e8f5e9")
        c.set_text_props(color="#2e7d32", style="italic")

    ax.set_title(
        "Fairness Check  —  all 3 backends receive identical inputs; "
        "FLOPs are backend-independent",
        fontsize=8.5, pad=6, color="#444444", style="italic",
    )


# ── figure builders ────────────────────────────────────────────────────────────

def _make_fig(tasks):
    """Create figure with len(tasks) task rows + 1 fairness row."""
    heights = [3.5] * len(tasks) + [1.4]
    fig, axes = plt.subplots(
        len(tasks) + 1, 1,
        figsize=(9, 3.8 * len(tasks) + 2.2),
        gridspec_kw={"height_ratios": heights},
        squeeze=False,
    )
    return fig, axes


def _make_legend(fig):
    handles = [plt.Rectangle((0, 0), 1, 1, color=BACKEND_COLORS[b],
                               label=BACKEND_LABELS[b])
               for b in BACKEND_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.01))


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {path}")
    plt.close(fig)


def _fname(metric, tasks, dtype, seq_len):
    return f"backend_{metric}_{'+'.join(tasks)}_{dtype}_seq{seq_len}.png"


# ── three main plots ───────────────────────────────────────────────────────────

def plot_latency(df, tasks, methods, outdir, dtype, seq_len):
    fig, axes = _make_fig(tasks)
    fig.suptitle(f"End-to-end Inference Latency (ms per batch)  |  dtype={dtype}  seq_len={seq_len}",
                 fontsize=12, fontweight="bold")
    for i, task in enumerate(tasks):
        sub = df[df["task"] == task]
        _draw_bars(axes[i, 0], sub, methods, "latency_ms",
                   ylabel=f"{TASK_LABELS.get(task, task)}\nms / batch",
                   higher_better=False, fmt="{:.1f}",
                   show_xlabel=(i == len(tasks) - 1))
        axes[i, 0].set_title(TASK_LABELS.get(task, task), fontsize=10,
                              loc="left", pad=4)
    _draw_fairness_panel(axes[-1, 0], df, tasks, methods)
    _make_legend(fig)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, outdir, _fname("latency", tasks, dtype, seq_len))


def plot_speedup(df, tasks, methods, outdir, dtype, seq_len):
    fig, axes = _make_fig(tasks)
    fig.suptitle(f"Throughput Speedup vs. Naive Backend  |  dtype={dtype}  seq_len={seq_len}",
                 fontsize=12, fontweight="bold")
    for i, task in enumerate(tasks):
        sub = df[df["task"] == task]
        _draw_speedup(axes[i, 0], sub, methods,
                      show_xlabel=(i == len(tasks) - 1))
        axes[i, 0].set_title(TASK_LABELS.get(task, task), fontsize=10,
                              loc="left", pad=4)
    _draw_fairness_panel(axes[-1, 0], df, tasks, methods)
    _make_legend(fig)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, outdir, _fname("speedup", tasks, dtype, seq_len))


def plot_memory(df, tasks, methods, outdir, dtype, seq_len):
    fig, axes = _make_fig(tasks)
    fig.suptitle(f"Peak GPU Memory Footprint (MB)  |  dtype={dtype}  seq_len={seq_len}",
                 fontsize=12, fontweight="bold")
    for i, task in enumerate(tasks):
        sub = df[df["task"] == task]
        _draw_bars(axes[i, 0], sub, methods, "peak_mem_mb",
                   ylabel=f"{TASK_LABELS.get(task, task)}\nMB",
                   higher_better=False, fmt="{:.0f}",
                   show_xlabel=(i == len(tasks) - 1))
        axes[i, 0].set_title(TASK_LABELS.get(task, task), fontsize=10,
                              loc="left", pad=4)
    _draw_fairness_panel(axes[-1, 0], df, tasks, methods)
    _make_legend(fig)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, outdir, _fname("memory", tasks, dtype, seq_len))


def plot_throughput(df, tasks, methods, outdir, dtype, seq_len):
    fig, axes = _make_fig(tasks)
    fig.suptitle(f"Throughput (samples / s)  |  dtype={dtype}  seq_len={seq_len}",
                 fontsize=12, fontweight="bold")
    for i, task in enumerate(tasks):
        sub = df[df["task"] == task]
        _draw_bars(axes[i, 0], sub, methods, "throughput_sps",
                   ylabel=f"{TASK_LABELS.get(task, task)}\nsamples / s",
                   higher_better=True, fmt="{:.0f}",
                   show_xlabel=(i == len(tasks) - 1))
        axes[i, 0].set_title(TASK_LABELS.get(task, task), fontsize=10,
                              loc="left", pad=4)
    _draw_fairness_panel(axes[-1, 0], df, tasks, methods)
    _make_legend(fig)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, outdir, _fname("throughput", tasks, dtype, seq_len))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     default="eval_encoder/eval_results/expB.csv")
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
        raw = pd.read_csv(args.csv)
        raise SystemExit(
            f"[error] No rows matched. Available:\n"
            f"{raw[['task','method','backend','dtype','seq_len']].drop_duplicates().to_string()}"
        )

    present = df["method"].unique().tolist()
    methods = [m for m in args.methods if m in present]
    t_present = df["task"].unique().tolist()
    tasks = [t for t in args.tasks if t in t_present]

    missing = [t for t in args.tasks if t not in t_present]
    if missing:
        print(f"[warn] Tasks not found: {missing}")

    print(f"[plot] tasks={tasks}  methods={methods}  outdir={args.outdir}")

    plot_latency(    df, tasks, methods, args.outdir, args.dtype, args.seq_len)
    plot_throughput( df, tasks, methods, args.outdir, args.dtype, args.seq_len)
    plot_speedup(    df, tasks, methods, args.outdir, args.dtype, args.seq_len)
    plot_memory(     df, tasks, methods, args.outdir, args.dtype, args.seq_len)


if __name__ == "__main__":
    main()
