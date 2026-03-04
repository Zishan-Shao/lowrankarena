#!/usr/bin/env python3
"""
Plot nsys kernel analysis figure from expD.csv.

Shows n_gemm_variants and kernel time breakdown (Triton vs GEMM vs other)
for SVD vs AdaSVD across naive and flashsvd15 backends.

Usage:
    python eval_encoder/scripts/plot_nsys_kernel.py \
        --csv   eval_encoder/eval_results/expD.csv \
        --outdir eval_encoder/eval_results/figures
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ── config ─────────────────────────────────────────────────────────────────────
# point name → (method_label, backend_label)
POINT_META = {
    "mnli_svd_naive":          ("SVD",    "Naive"),
    "mnli_svd_flashsvd":       ("SVD",    "FlashSVD 1.0"),
    "mnli_svd_flashsvd15":     ("SVD",    "FlashSVD 1.5"),
    "mnli_adasvd_naive":       ("AdaSVD", "Naive"),
    "mnli_adasvd_flashsvd":    ("AdaSVD", "FlashSVD 1.0"),
    "mnli_adasvd_flashsvd15":  ("AdaSVD", "FlashSVD 1.5"),
}
BACKEND_COLORS = {"Naive": "#9e9e9e", "FlashSVD 1.0": "#ffa726", "FlashSVD 1.5": "#ef5350"}
METHOD_COLORS  = {"SVD": "#5c85d6", "AdaSVD": "#e06c5a"}

# stacked bar colors for kernel time breakdown
C_TRITON  = "#ef5350"   # FlashSVD Triton fused kernels
C_GEMM    = "#42a5f5"   # cuBLAS / CUTLASS GEMM
C_OTHER   = "#bdbdbd"   # elementwise, layer norm, etc.


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ["n_gemm_variants", "triton_time_pct", "gemm_time_pct",
                "memcpy_time_pct", "sync_time_pct", "top1_time_pct",
                "n_unique_kernels", "total_kernel_calls"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot(df, outdir):
    # order: naive, flashsvd 1.0, flashsvd 1.5; SVD before AdaSVD
    order = ["mnli_svd_naive", "mnli_svd_flashsvd", "mnli_svd_flashsvd15",
             "mnli_adasvd_naive", "mnli_adasvd_flashsvd", "mnli_adasvd_flashsvd15"]
    rows = []
    for pt in order:
        r = df[df["point"] == pt]
        if len(r) == 0:
            print(f"[warn] point not found: {pt}")
            continue
        rows.append(r.iloc[0])

    if not rows:
        raise SystemExit("[error] No matching rows found in CSV.")

    labels   = [f"{POINT_META[r['point']][0]}\n{POINT_META[r['point']][1]}"
                for r in rows]
    x        = np.arange(len(rows))
    bar_w    = 0.55

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Kernel-Level Analysis  |  MNLI  bf16  seq=512\n"
                 "(nsys cuda_gpu_kern_sum)",
                 fontsize=11, fontweight="bold")

    # ── Panel 1: n_gemm_variants ──────────────────────────────────────────────
    colors1 = [BACKEND_COLORS[POINT_META[r["point"]][1]] for r in rows]
    vals1   = [r["n_gemm_variants"] for r in rows]
    bars1   = ax1.bar(x, vals1, width=bar_w, color=colors1, zorder=3,
                      edgecolor="white", linewidth=0.8)
    ymax1   = max(vals1) * 1.25
    for rect, v in zip(bars1, vals1):
        ax1.text(rect.get_x() + rect.get_width() / 2,
                 rect.get_height() + ymax1 * 0.02,
                 str(int(v)), ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="#222222")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("# Distinct GEMM Kernel Configs", fontsize=9)
    ax1.set_title("GEMM Kernel Diversity\n(higher → more rank heterogeneity)",
                  fontsize=10, fontweight="bold")
    ax1.set_ylim(0, ymax1)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax1.set_axisbelow(True)

    # vertical separator between SVD and AdaSVD groups
    ax1.axvline(2.5, color="#cccccc", linestyle="--", linewidth=1, zorder=2)
    ax1.text(1.0, ymax1 * 0.97, "SVD", ha="center", va="top",
             fontsize=8, color="#555555", style="italic")
    ax1.text(4.0, ymax1 * 0.97, "AdaSVD", ha="center", va="top",
             fontsize=8, color="#555555", style="italic")

    # ── Panel 2: kernel time breakdown (stacked bars) ─────────────────────────
    triton_pct = [r["triton_time_pct"] for r in rows]
    gemm_pct   = [r["gemm_time_pct"]   for r in rows]
    other_pct  = [max(0.0, 100.0 - r["triton_time_pct"] - r["gemm_time_pct"])
                  for r in rows]

    b_triton = ax2.bar(x, triton_pct, width=bar_w, color=C_TRITON,
                       label="Triton fused\n(FlashSVD)", zorder=3)
    b_gemm   = ax2.bar(x, gemm_pct, width=bar_w, bottom=triton_pct,
                       color=C_GEMM, label="cuBLAS/CUTLASS\nGEMM", zorder=3)
    bottom2  = [t + g for t, g in zip(triton_pct, gemm_pct)]
    b_other  = ax2.bar(x, other_pct, width=bar_w, bottom=bottom2,
                       color=C_OTHER, label="Other\n(elementwise, LN…)", zorder=3)

    # annotate total GEMM+Triton %
    for i, r in enumerate(rows):
        total = r["triton_time_pct"] + r["gemm_time_pct"]
        ax2.text(x[i], 101, f"{total:.0f}%", ha="center", va="bottom",
                 fontsize=8, color="#333333")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("GPU Time (%)", fontsize=9)
    ax2.set_ylim(0, 115)
    ax2.set_title("Kernel Time Breakdown\n(stacked, per-point)",
                  fontsize=10, fontweight="bold")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax2.set_axisbelow(True)
    ax2.axvline(2.5, color="#cccccc", linestyle="--", linewidth=1, zorder=2)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fname = "nsys_kernel_analysis_mnli_bf16_seq512.png"
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",    default="eval_encoder/eval_results/expD.csv")
    p.add_argument("--outdir", default="eval_encoder/eval_results/figures")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"[error] File not found: {args.csv}")

    df = load(args.csv)
    plot(df, args.outdir)


if __name__ == "__main__":
    main()
