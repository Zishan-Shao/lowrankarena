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
    # - seq_pad_pct must come from real rows (synthetic always = 0, meaningless for right panel).
    # Keep one row per method: prefer real, fall back to synthetic.
    df_real  = df[df.get("input_mode", pd.Series(dtype=str)) == "real"]
    df_synth = df[df.get("input_mode", pd.Series(dtype=str)) == "synthetic"]
    if "input_mode" in df.columns and not df_real.empty:
        # merge: take flops cols from real row (seq_pad_pct is meaningful there)
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

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12, 5.5),
        gridspec_kw={"wspace": 0.35},
    )
    task_label = TASK_LABELS.get(task, task)
    fig.suptitle(
        f"FLOPs Breakdown  |  {task_label}  dtype={dtype}  seq_len={seq_len}",
        fontsize=12, fontweight="bold",
    )

    # ── Panel 1: compute quality (useful vs rank-pad, normalized to total_flops) ──
    ax_left.set_title(
        "Compute Quality\n(excl. sequence-padding tokens)",
        fontsize=10, fontweight="bold",
    )
    for xi, method in enumerate(methods):
        row = df[df["method"] == method].iloc[0]
        rpad_pct   = float(row["rank_pad_pct"])
        useful_pct = 100.0 - rpad_pct

        b_u = ax_left.bar(xi, useful_pct, BAR_W, color=C_USEFUL,   zorder=3)
        b_r = ax_left.bar(xi, rpad_pct,   BAR_W, color=C_RANKPAD,
                          bottom=useful_pct, zorder=3)
        _annotate(ax_left, b_u, useful_pct,  bottom=0.0,        color="white")
        _annotate(ax_left, b_r, rpad_pct,    bottom=useful_pct, color="white", min_pct=1.0)

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(xlabels, fontsize=10)
    ax_left.set_ylabel("Proportion of total FLOPs (%)", fontsize=9)
    ax_left.set_ylim(0, 108)
    ax_left.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax_left.set_axisbelow(True)
    ax_left.set_yticks(range(0, 101, 20))

    # no per-panel legend on the left; a single shared legend sits right of the right panel

    # ── Panel 2: full inference budget (including seq-padding) ────────────────
    ax_right.set_title(
        "Full Inference Budget\n(incl. sequence-padding tokens)",
        fontsize=10, fontweight="bold",
    )
    for xi, method in enumerate(methods):
        row = df[df["method"] == method].iloc[0]
        total      = float(row["total_flops_abs"])
        useful_abs = float(row["useful_flops_abs"])
        rpad_abs   = float(row["rank_pad_flops_abs"])
        seq_pad_f  = float(row["seq_pad_pct"]) / 100.0   # fraction

        # split each component into seq-content portion vs seq-pad portion
        content_f = 1.0 - seq_pad_f
        eff_useful  = 100.0 * (useful_abs / total) * content_f
        eff_rpad    = 100.0 * (rpad_abs   / total) * content_f
        eff_seqpad  = 100.0 * seq_pad_f

        b_s = ax_right.bar(xi, eff_seqpad, BAR_W, color=C_SEQPAD,  zorder=3)
        b_r = ax_right.bar(xi, eff_rpad,   BAR_W, color=C_RANKPAD,
                           bottom=eff_seqpad, zorder=3)
        b_u = ax_right.bar(xi, eff_useful, BAR_W, color=C_USEFUL,
                           bottom=eff_seqpad + eff_rpad, zorder=3)
        _annotate(ax_right, b_s, eff_seqpad, bottom=0.0,                    color="#555555")
        _annotate(ax_right, b_r, eff_rpad,   bottom=eff_seqpad,            color="white", min_pct=0.4)
        _annotate(ax_right, b_u, eff_useful, bottom=eff_seqpad + eff_rpad, color="white")

    ax_right.set_xticks(x)
    ax_right.set_xticklabels(xlabels, fontsize=10)
    ax_right.set_ylabel("Proportion of total FLOPs (%)", fontsize=9)
    ax_right.set_ylim(0, 108)
    ax_right.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax_right.set_axisbelow(True)
    ax_right.set_yticks(range(0, 101, 20))

    # Single shared legend to the right of the right panel — covers all 3 colors used in both panels
    ax_right.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=C_USEFUL,  label="Effective / useful compute"),
            plt.Rectangle((0, 0), 1, 1, color=C_RANKPAD, label="Rank-alignment overhead\n(Triton next_pow2(R))"),
            plt.Rectangle((0, 0), 1, 1, color=C_SEQPAD,  label="Sequence-padding tokens\n(input-level waste)"),
        ],
        loc="upper left", bbox_to_anchor=(1.03, 1),
        fontsize=8.5, framealpha=0.9, borderaxespad=0,
    )

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(outdir, exist_ok=True)
    fname = f"flops_breakdown_{task}_{dtype}_seq{seq_len}.png"
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
