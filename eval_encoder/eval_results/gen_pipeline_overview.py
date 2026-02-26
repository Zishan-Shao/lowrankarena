#!/usr/bin/env python3
"""
Encoder Benchmark Pipeline Overview Figure
Output: eval_encoder/eval_results/figures/pipeline_overview.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── canvas ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 13))
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 20)
ax.set_ylim(0, 11)
ax.set_facecolor("#FAFAFA")
ax.axis("off")

# ── color scheme ─────────────────────────────────────────────────────────────
CLR = {
    "input":      ("#E3F2FD", "#1565C0"),
    "compress":   ("#FFF8E1", "#E65100"),
    "accuracy":   ("#E8F5E9", "#2E7D32"),
    "efficiency": ("#FBE9E7", "#BF360C"),
    "output":     ("#F3E5F5", "#6A1B9A"),
    "arrow":      "#546E7A",
}

# ── helpers ──────────────────────────────────────────────────────────────────
def draw_block(x, y, w, h, kind, title, lines, title_fs=15, body_fs=14):
    fc, ec = CLR[kind]
    # shadow
    ax.add_patch(FancyBboxPatch((x+0.07, y-0.07), w, h,
                                boxstyle="round,pad=0.18",
                                facecolor="#CCCCCC", edgecolor="none",
                                zorder=2, alpha=0.5))
    # main box
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.18",
                                facecolor=fc, edgecolor=ec,
                                linewidth=2.0, zorder=3))
    # title bar (solid rect inside top of box)
    bar_h = 0.65
    ax.add_patch(mpatches.FancyBboxPatch((x+0.05, y+h-bar_h-0.05), w-0.10, bar_h,
                                          boxstyle="round,pad=0.08",
                                          facecolor=ec, edgecolor="none",
                                          zorder=4, alpha=0.90))
    ax.text(x + w/2, y + h - bar_h/2 - 0.05, title,
            ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color="white", zorder=5)
    # body lines
    usable_h = h - bar_h - 0.25
    step     = usable_h / max(len(lines), 1)
    for i, ln in enumerate(lines):
        bold   = ln.startswith("**")
        italic = ln.startswith("//")
        txt    = ln.lstrip("*").lstrip("/").strip()
        ypos   = y + h - bar_h - 0.22 - (i + 0.5) * step
        ax.text(x + w/2, ypos, txt,
                ha="center", va="center",
                fontsize=body_fs,
                fontweight="bold" if bold else "normal",
                fontstyle="italic" if italic else "normal",
                color="#1A1A1A" if txt else "#FFFFFF",
                zorder=5)

def arrow(x1, y1, x2, y2, rad=0.0, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>",
                                color=CLR["arrow"], lw=2.0,
                                mutation_scale=18,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=6)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.18, label,
                ha="center", va="bottom", fontsize=8.5,
                color=CLR["arrow"], fontstyle="italic", zorder=7)

# ── block positions (x, y, w, h) ─────────────────────────────────────────────
IN  = (0.30, 1.8, 3.3, 7.4)
CO  = (4.20, 1.8, 3.8, 7.4)
AC  = (8.80, 5.5, 4.6, 3.7)
EF  = (8.80, 1.8, 4.6, 3.4)
OUT = (14.20, 1.8, 5.5, 7.4)

# ── Block 1 : Input ──────────────────────────────────────────────────────────
draw_block(*IN, "input", "Input Models",
           ["Task-finetuned BERT-base",
            "//textattack / howey (HuggingFace)",
            "",
            "**8 GLUE Tasks**",
            "CoLA · SST-2 · MRPC · QQP",
            "MNLI · QNLI · RTE · STS-B",
            "",
            "seq_len : 128 / 256 / 512",
            "batch_size : 8 / 16 / 32 / 64",
            "dtype : fp32 / bf16",
            ])

# ── Block 2 : Compression ────────────────────────────────────────────────────
draw_block(*CO, "compress", "Compression Methods",
           ["**SVD**   plain torch.linalg.svd",
            "**FWSVD**   Fisher-weighted SVD",
            "**DRONE**   covariance calibration",
            "**AdaSVD**   ARS adaptive rank",
            "",
            "Mode A — per-head",
            "ra48 / rf256 / rw208",
            "",
            "Mode B — full-matrix",
            "ra312 / rf256 / rw208",
            "",
            "param_ratio ≈ 0.527  (equal)",
            ])

# ── Block 3a : Accuracy ──────────────────────────────────────────────────────
draw_block(*AC, "accuracy", "Accuracy Evaluation",
           ["//glue_pipeline.py",
            "**Stage 1**  compress → eval (no finetune)",
            "**Stage 2**  compress → finetune 3 ep → eval",
            "",
            "Metrics: MCC / Acc / F1 / Pearson",
            "G-AVG (8 tasks)  ·  A-AVG (4 Acc tasks)",
            ])

# ── Block 3b : Efficiency ────────────────────────────────────────────────────
draw_block(*EF, "efficiency", "Efficiency Evaluation",
           ["//run_encoder_benchmark  /  bench_synthetic",
            "3 Backends: einsum · SDPA · FlashSVD",
            "",
            "Sweeps: seq-len (128/256/512)",
            "        batch-size (8/16/32/64)",
            "        dtype (fp32 / bf16)",
            "",
            "Metrics: peak_mem_infer_mb",
            "         throughput_sps  ·  latency_ms",
            ],
           title_fs=13)

# ── Block 4 : Output ─────────────────────────────────────────────────────────
draw_block(*OUT, "output", "Results",
           ["**Accuracy**",
            "Stage1 / Stage2 GLUE tables",
            "per-head vs full-matrix analysis",
            "MRPC collapse / recovery study",
            "",
            "**Efficiency  (14 figures)**",
            "Kernel-tier memory & throughput",
            "Memory breakdown (param / activ)",
            "Seq-len scaling  −32% → −65%",
            "dtype × backend scaling",
            "Batch-size scaling",
            "",
            "**Accuracy–Memory Pareto**",
            "FlashSVD: −1296 MB, same accuracy",
            ])

# ── Arrows ───────────────────────────────────────────────────────────────────
# Input → Compression
arrow(IN[0]+IN[2],  IN[1]+IN[3]/2,
      CO[0],        CO[1]+CO[3]/2)

# Compression → Accuracy (fork up)
arrow(CO[0]+CO[2],  CO[1]+CO[3]*0.72,
      AC[0],        AC[1]+AC[3]/2,   rad=-0.22)

# Compression → Efficiency (fork down)
arrow(CO[0]+CO[2],  CO[1]+CO[3]*0.28,
      EF[0],        EF[1]+EF[3]/2,   rad=0.22)

# Accuracy → Output
arrow(AC[0]+AC[2],  AC[1]+AC[3]/2,
      OUT[0],       OUT[1]+OUT[3]*0.72, rad=-0.15)

# Efficiency → Output
arrow(EF[0]+EF[2],  EF[1]+EF[3]/2,
      OUT[0],       OUT[1]+OUT[3]*0.28, rad=0.15)

# ── fork dot ─────────────────────────────────────────────────────────────────
fx = CO[0]+CO[2] + 0.30
ax.plot(fx, CO[1]+CO[3]/2, "o", color=CLR["arrow"], ms=7, zorder=7)

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(10, 10.55, "Encoder Benchmark Pipeline",
        ha="center", va="center", fontsize=22, fontweight="bold", color="#212121")
ax.text(10, 10.12,
        "Low-rank SVD Compression  ×  Backend Evaluation  ×  GLUE Accuracy",
        ha="center", va="center", fontsize=15, color="#555555")

# ── legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=CLR["input"][0],      edgecolor=CLR["input"][1],      label="Input"),
    mpatches.Patch(facecolor=CLR["compress"][0],   edgecolor=CLR["compress"][1],   label="Compression"),
    mpatches.Patch(facecolor=CLR["accuracy"][0],   edgecolor=CLR["accuracy"][1],   label="Accuracy Track"),
    mpatches.Patch(facecolor=CLR["efficiency"][0], edgecolor=CLR["efficiency"][1], label="Efficiency Track"),
    mpatches.Patch(facecolor=CLR["output"][0],     edgecolor=CLR["output"][1],     label="Results"),
]
ax.legend(handles=legend_items, loc="lower center", ncol=5,
          fontsize=13, framealpha=0.9,
          bbox_to_anchor=(0.5, 0.01))

# ── save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "pipeline_overview.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#FAFAFA")
print(f"Saved: {out_path}")
plt.close()
