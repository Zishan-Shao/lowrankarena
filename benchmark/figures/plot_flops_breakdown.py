#!/usr/bin/env python3
"""
FLOPs breakdown stacked bar chart.

Two panels side-by-side:
  Left  — Compute quality (useful vs. rank-alignment overhead, excl. seq padding)
  Right — Full inference budget (seq-padding overhead + rank-pad + effective compute)

Usage:
    python eval_encoder/scripts/plot_flops_breakdown.py \
        --csv     eval_encoder/eval_results/expB.csv \
        --task    mnli \
        --dtype   bf16 \
        --seq_len 512 \
        --outdir  eval_encoder/eval_results/figures
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER  = ["svd", "fwsvd", "drone", "adasvd"]
METHOD_LABELS = {"svd": "SVD", "fwsvd": "FWSVD", "drone": "DRONE", "adasvd": "AdaSVD"}
TASK_LABELS   = {"mnli": "MNLI", "cola": "CoLA", "stsb": "STS-B",
                 "sst2": "SST-2", "mrpc": "MRPC"}

# ── colors ─────────────────────────────────────────────────────────────────────
C_USEFUL  = "#43a047"   # green  — effective computation
C_RANKPAD = "#fb8c00"   # orange — Triton rank-alignment overhead
C_SEQPAD  = "#e0e0e0"   # gray   — sequence-padding tokens (input-level waste)

BAR_W = 0.55


def load(path, task, dtype, seq_len):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[
        (df["task"] == task) &
        (df["dtype"] == dtype) &
        (df["seq_len"].astype(int) == seq_len) &
        (df["backend"] == "naive")      # FLOPs are backend-invariant; use naive
    ].copy()
    for col in ["useful_flops_abs", "rank_pad_flops_abs", "total_flops_abs",
                "rank_pad_pct", "seq_pad_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Separate real vs synthetic rows.
    # - FLOPs fields (rank_pad_pct etc.) are static — same for both input_modes.
    # - seq_pad_pct: synthetic=0% (fully-utilized input), real=~92% (MNLI fixed seq=512).
    # Prefer synthetic: isolates rank-pad overhead without seq-padding confound;
    # consistent with left panel which already excludes seq-padding.
    # Fall back to real if synthetic not present.
    df_synth = df[df.get("input_mode", pd.Series(dtype=str)) == "synthetic"]
    df_real  = df[df.get("input_mode", pd.Series(dtype=str)) == "real"]
    if "input_mode" in df.columns and not df_synth.empty:
        df = df_synth.drop_duplicates(subset=["method"])
    elif not df_real.empty:
        df = df_real.drop_duplicates(subset=["method"])
    else:
        df = df.drop_duplicates(subset=["method"])
    return df


def _annotate(ax, bars, pct, bottom=0.0, color="#333333", min_pct=1.5):
    """Place percentage label inside bar if tall enough. bars = BarContainer."""
    if pct < min_pct:
        return
    rect = bars[0]
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        bottom + pct / 2,
        f"{pct:.1f}%",
        ha="center", va="center",
        fontsize=8.5, color=color, fontweight="bold",
    )


def plot(df, task, dtype, seq_len, outdir):
    methods = [m for m in METHOD_ORDER if m in df["method"].values]
    xlabels = [METHOD_LABELS.get(m, m) for m in methods]
    x = np.arange(len(methods))

    task_label = TASK_LABELS.get(task, task)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.set_title(
        f"FLOPs Breakdown  |  {task_label}  dtype={dtype}  seq_len={seq_len}",
        fontsize=11, fontweight="bold", pad=10,
    )

    for xi, method in enumerate(methods):
        row = df[df["method"] == method].iloc[0]
        rpad_pct   = float(row["rank_pad_pct"])
        useful_pct = 100.0 - rpad_pct

        b_u = ax.bar(xi, useful_pct, BAR_W, color=C_USEFUL,  zorder=3)
        b_r = ax.bar(xi, rpad_pct,   BAR_W, color=C_RANKPAD,
                     bottom=useful_pct, zorder=3)
        _annotate(ax, b_u, useful_pct,  bottom=0.0,        color="white")
        _annotate(ax, b_r, rpad_pct,    bottom=useful_pct, color="white", min_pct=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_ylabel("Proportion of total FLOPs (%)", fontsize=10)
    ax.set_ylim(0, 108)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(range(0, 101, 20))

    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=C_USEFUL,  label="Effective / useful compute"),
            plt.Rectangle((0, 0), 1, 1, color=C_RANKPAD, label="Rank-alignment overhead\n(Triton next_pow2(R))"),
        ],
        loc="upper left", bbox_to_anchor=(1.02, 1),
        fontsize=9, framealpha=0.9, borderaxespad=0,
    )

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(outdir, exist_ok=True)
    fname = "fig13_flops_breakdown.png"
    out   = os.path.join(outdir, fname)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     default="eval_encoder/eval_results/expB.csv")
    p.add_argument("--task",    default="mnli")
    p.add_argument("--dtype",   default="bf16")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--outdir",  default="eval_encoder/eval_results/figures")
    args = p.parse_args()

    df = load(args.csv, args.task, args.dtype, args.seq_len)
    if df.empty:
        raise SystemExit(f"[error] No rows found for task={args.task} dtype={args.dtype} seq_len={args.seq_len}")

    plot(df, args.task, args.dtype, args.seq_len, args.outdir)


if __name__ == "__main__":
    main()
