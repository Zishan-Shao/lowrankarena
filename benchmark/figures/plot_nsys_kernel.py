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
    "mnli_svd_flashsvd":       ("SVD",    "Flash 1.0"),
    "mnli_svd_flashsvd15":     ("SVD",    "Flash 1.5"),
    "mnli_adasvd_naive":       ("AdaSVD", "Naive"),
    "mnli_adasvd_flashsvd":    ("AdaSVD", "Flash 1.0"),
    "mnli_adasvd_flashsvd15":  ("AdaSVD", "Flash 1.5"),
}
BACKEND_COLORS = {"Naive": "#9e9e9e", "Flash 1.0": "#42a5f5", "Flash 1.5": "#ef5350"}
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

    os.makedirs(outdir, exist_ok=True)

    # ── Figure A: n_gemm_variants ─────────────────────────────────────────────
    fig_a, ax1 = plt.subplots(figsize=(6.5, 4.5))
    fig_a.suptitle("Rank Heterogeneity Improves Kernel Utilization"
                   "  (MNLI, bf16, seq=512)",
                   fontsize=11, fontweight="bold")

    colors1 = [BACKEND_COLORS[POINT_META[r["point"]][1]] for r in rows]
    vals1   = [r["n_gemm_variants"] for r in rows]
    bars1   = ax1.bar(x, vals1, width=bar_w, color=colors1, zorder=3,
                      edgecolor="white", linewidth=0.8)
    ymax1   = 15   # max observed is 14; fixed range makes gap clearer
    for rect, v in zip(bars1, vals1):
        ax1.text(rect.get_x() + rect.get_width() / 2,
                 rect.get_height() + 0.2,
                 str(int(v)), ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="#222222")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("# Distinct GEMM Kernel Configs", fontsize=9)
    ax1.set_title("GEMM Kernel Diversity\n(AdaSVD heterogeneous ranks → more configs)",
                  fontsize=10, fontweight="bold")
    ax1.set_ylim(0, ymax1)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax1.set_axisbelow(True)

    ax1.axvline(2.5, color="#cccccc", linestyle="--", linewidth=1, zorder=2)
    ax1.text(1.0, ymax1 * 0.96, "Uniform ranks", ha="center", va="top",
             fontsize=8, color="#555555", style="italic")
    ax1.text(4.0, ymax1 * 0.96, "Heterogeneous ranks", ha="center", va="top",
             fontsize=8, color="#555555", style="italic")

    legend_patches = [mpatches.Patch(color=BACKEND_COLORS[b], label=b)
                      for b in ["Naive", "Flash 1.0", "Flash 1.5"]]
    ax1.legend(handles=legend_patches, loc="upper left", fontsize=8, framealpha=0.9)

    fig_a.tight_layout()
    out_a = os.path.join(outdir, "fig15_nsys_kernel_A.png")
    fig_a.savefig(out_a, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out_a}")
    plt.close(fig_a)

    # ── Figure B: kernel time breakdown (stacked bars) ────────────────────────
    triton_pct = [r["triton_time_pct"] for r in rows]
    gemm_pct   = [r["gemm_time_pct"]   for r in rows]
    other_pct  = [max(0.0, 100.0 - r["triton_time_pct"] - r["gemm_time_pct"])
                  for r in rows]

    HIGHLIGHT = {i for i in {4, 5} if i < len(rows)}

    fig_b, ax2 = plt.subplots(figsize=(6.5, 4.5))
    fig_b.suptitle("Rank Heterogeneity Improves Kernel Utilization"
                   "  (MNLI, bf16, seq=512)",
                   fontsize=11, fontweight="bold")

    b_triton = ax2.bar(x, triton_pct, width=bar_w, color=C_TRITON,
                       label="Triton fused\n(FlashSVD)", zorder=3)
    b_gemm   = ax2.bar(x, gemm_pct, width=bar_w, bottom=triton_pct,
                       color=C_GEMM, label="cuBLAS/CUTLASS\nGEMM", zorder=3)
    bottom2  = [t + g for t, g in zip(triton_pct, gemm_pct)]
    b_other  = ax2.bar(x, other_pct, width=bar_w, bottom=bottom2,
                       color=C_OTHER, label="Other\n(elementwise, LN…)", zorder=3)

    for i in HIGHLIGHT:
        total_h = triton_pct[i] + gemm_pct[i] + other_pct[i]
        rect = mpatches.Rectangle((x[i] - bar_w / 2, 0), bar_w, total_h,
                                   linewidth=2.2, edgecolor="#111111",
                                   facecolor="none", zorder=5)
        ax2.add_patch(rect)
        ax2.text(x[i], total_h + 1.5, "★", ha="center", va="bottom",
                 fontsize=11, color="#111111", zorder=6)

    for i, t in enumerate(triton_pct):
        if t >= 8:
            label = f"{t:.0f}%\nTriton"
            ax2.text(x[i], t / 2, label, ha="center", va="center",
                     fontsize=7.5, color="white", fontweight="bold", zorder=4)
        elif t > 0:
            ax2.text(x[i], t + 1, f"{t:.0f}%", ha="center", va="bottom",
                     fontsize=7.5, color=C_TRITON, fontweight="bold", zorder=4)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("GPU Time (%)", fontsize=9)
    ax2.set_ylim(0, 115)
    ax2.set_title("Kernel Time Breakdown\n(Triton share ↑ = better kernel utilization)",
                  fontsize=10, fontweight="bold")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax2.set_axisbelow(True)
    ax2.axvline(2.5, color="#cccccc", linestyle="--", linewidth=1, zorder=2)
    ax2.text(1.0, 113, "Uniform ranks", ha="center", va="top",
             fontsize=8, color="#555555", style="italic")
    ax2.text(4.0, 113, "Heterogeneous ranks", ha="center", va="top",
             fontsize=8, color="#555555", style="italic")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1),
               borderaxespad=0, fontsize=8.5, framealpha=0.9)

    fig_b.tight_layout()
    out_b = os.path.join(outdir, "fig15_nsys_kernel_B.png")
    fig_b.savefig(out_b, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out_b}")
    plt.close(fig_b)

    # ── Combined fig15: 1×2 Kernel Utilization Analysis ──────────────────────
    fig_c, (cx0, cx1) = plt.subplots(1, 2, figsize=(13.0, 4.5),
                                      gridspec_kw={"wspace": 0.40})

    # Panel A — GEMM Kernel Diversity
    colors1 = [BACKEND_COLORS[POINT_META[r["point"]][1]] for r in rows]
    vals1   = [r["n_gemm_variants"] for r in rows]
    bars1   = cx0.bar(x, vals1, width=bar_w, color=colors1, zorder=3,
                      edgecolor="white", linewidth=0.8)
    ymax1   = 15
    for rect, v in zip(bars1, vals1):
        cx0.text(rect.get_x() + rect.get_width() / 2,
                 rect.get_height() + 0.2,
                 str(int(v)), ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="#222222")
    cx0.set_xticks(x)
    cx0.set_xticklabels(labels, fontsize=8)
    cx0.set_ylabel("# Distinct GEMM Kernel Configs", fontsize=9)
    cx0.set_title("GEMM Kernel Diversity", fontweight='bold')
    cx0.set_ylim(0, ymax1)
    cx0.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    cx0.set_axisbelow(True)
    cx0.axvline(2.5, color="#cccccc", linestyle="--", linewidth=1, zorder=2)
    cx0.text(1.0, ymax1 * 0.96, "Uniform ranks", ha="center", va="top",
             fontsize=8, color="#555555", style="italic")
    cx0.text(4.0, ymax1 * 0.96, "Heterogeneous ranks", ha="center", va="top",
             fontsize=8, color="#555555", style="italic")

    # Panel B — Kernel Time Breakdown
    b_triton = cx1.bar(x, triton_pct, width=bar_w, color=C_TRITON,
                       label="Triton fused (FlashSVD)", zorder=3)
    b_gemm   = cx1.bar(x, gemm_pct, width=bar_w, bottom=triton_pct,
                       color=C_GEMM, label="cuBLAS/CUTLASS GEMM", zorder=3)
    bottom2  = [t + g for t, g in zip(triton_pct, gemm_pct)]
    cx1.bar(x, other_pct, width=bar_w, bottom=bottom2,
            color=C_OTHER, label="Other (elementwise, LN…)", zorder=3)
    for i in HIGHLIGHT:
        total_h = triton_pct[i] + gemm_pct[i] + other_pct[i]
        rect = mpatches.Rectangle((x[i] - bar_w / 2, 0), bar_w, total_h,
                                   linewidth=2.2, edgecolor="#111111",
                                   facecolor="none", zorder=5)
        cx1.add_patch(rect)
        cx1.text(x[i], total_h + 1.5, "★", ha="center", va="bottom",
                 fontsize=11, color="#111111", zorder=6)
    for i, t in enumerate(triton_pct):
        if t >= 8:
            cx1.text(x[i], t / 2, f"{t:.0f}%\nTriton", ha="center", va="center",
                     fontsize=7.5, color="white", fontweight="bold", zorder=4)
        elif t > 0:
            cx1.text(x[i], t + 1, f"{t:.0f}%", ha="center", va="bottom",
                     fontsize=7.5, color=C_TRITON, fontweight="bold", zorder=4)
    cx1.set_xticks(x)
    cx1.set_xticklabels(labels, fontsize=8)
    cx1.set_ylabel("GPU Time (%)", fontsize=9)
    cx1.set_ylim(0, 115)
    cx1.set_title("Kernel Time Breakdown", fontweight='bold')
    cx1.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    cx1.set_axisbelow(True)
    cx1.axvline(2.5, color="#cccccc", linestyle="--", linewidth=1, zorder=2)
    cx1.text(1.0, 113, "Uniform ranks", ha="center", va="top",
             fontsize=8, color="#555555", style="italic")
    cx1.text(4.0, 113, "Heterogeneous ranks", ha="center", va="top",
             fontsize=8, color="#555555", style="italic")

    # Shared legend: backend patches (panel A) + kernel type (panel B)
    leg_backend = [mpatches.Patch(color=BACKEND_COLORS[b], label=b)
                   for b in ["Naive", "Flash 1.0", "Flash 1.5"]]
    handles_b, labels_b = cx1.get_legend_handles_labels()
    all_handles = leg_backend + handles_b
    all_labels  = ["Naive", "Flash 1.0", "Flash 1.5"] + labels_b
    fig_c.legend(all_handles, all_labels, loc='lower center',
                 ncol=len(all_handles), framealpha=0.9, fontsize=10.5,
                 bbox_to_anchor=(0.5, -0.08))
    fig_c.text(0.01, 0.5, "Kernel Utilization Analysis", rotation=90,
               va='center', ha='center', fontsize=13, fontweight='bold')
    fig_c.suptitle("Rank Heterogeneity Improves Kernel Utilization  (MNLI, bf16, seq=512)",
                   fontsize=11, fontweight="bold")
    fig_c.tight_layout(rect=[0.03, 0.0, 1.0, 0.93])
    out_c = os.path.join(outdir, "fig15_nsys_combined.png")
    fig_c.savefig(out_c, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out_c}")
    plt.close(fig_c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",    default="experiments/results/expD_mnli_bf16_s512_b32.csv")
    p.add_argument("--outdir", default="experiments/figs/figures")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"[error] File not found: {args.csv}")

    df = load(args.csv)
    plot(df, args.outdir)


if __name__ == "__main__":
    main()
