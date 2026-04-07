"""
plot_eval_results.py — LowRankArena decoder benchmark visualisation.

Sources
  evaluate/results/llama31_8b.csv          (PPL + avg_score)
  evaluate/results/json/Llama-3.1-8B_*.json (per-task acc + stderr)

Output  evaluate/results/figures/
  fig1_ppl.png          — WikiText-2 PPL vs keep_ratio (log scale)
  fig2_per_method.png   — per-task accuracy per method (2×3 grid, error bands)
  fig3_avg_score.png    — avg zero-shot accuracy per method
  table1_results.tex    — full LaTeX table
"""
import csv, json, math, re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
HERE  = Path(__file__).parent
JDIR  = HERE / "json"
OUT   = HERE / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "legend.fontsize":   8,
})

# ── method config ─────────────────────────────────────────────────────────────
METHODS_ORDER = [
    "ASVD_asvd_raw",
    "Basis_sharing_basis_sharing",
    "SVDLLMv1_whitening_only",
    "SVDLLMv1_whitening_then_update",
    "SVDLLMv2_v2",
    "DobiSVD_cuij_Llama_3_1_8B_DobiSVD",
]
LABEL = {
    "ASVD_asvd_raw":                        "ASVD",
    "Basis_sharing_basis_sharing":           "Basis Sharing",
    "SVDLLMv1_whitening_only":              "SVD-LLM v1\n(whitening only)",
    "SVDLLMv1_whitening_then_update":       "SVD-LLM v1\n(whitening+update)",
    "SVDLLMv2_v2":                          "SVD-LLM v2",
    "DobiSVD_cuij_Llama_3_1_8B_DobiSVD":   "DobiSVD",
}
LABEL_SHORT = {k: v.replace("\n", " ") for k, v in LABEL.items()}
COLOR = {
    "ASVD_asvd_raw":                        "#e41a1c",
    "Basis_sharing_basis_sharing":           "#ff7f00",
    "SVDLLMv1_whitening_only":              "#4daf4a",
    "SVDLLMv1_whitening_then_update":       "#984ea3",
    "SVDLLMv2_v2":                          "#377eb8",
    "DobiSVD_cuij_Llama_3_1_8B_DobiSVD":   "#a65628",
}
MARKER = {
    "ASVD_asvd_raw":                        "o",
    "Basis_sharing_basis_sharing":           "s",
    "SVDLLMv1_whitening_only":              "^",
    "SVDLLMv1_whitening_then_update":       "D",
    "SVDLLMv2_v2":                          "v",
    "DobiSVD_cuij_Llama_3_1_8B_DobiSVD":   "P",
}
FALSE_PPL_METHODS = {"SVDLLMv1_whitening_then_update"}

# ── tasks shown in per-method subplot ─────────────────────────────────────────
TASKS = ["boolq", "hellaswag", "arc_challenge", "piqa", "winogrande", "openbookqa", "arc_easy", "mathqa"]
TASK_KEY = {   # JSON key → (acc key, stderr key)
    "boolq":         ("acc,none",      "acc_stderr,none"),
    "hellaswag":     ("acc_norm,none", "acc_norm_stderr,none"),
    "arc_challenge": ("acc_norm,none", "acc_norm_stderr,none"),
    "arc_easy":      ("acc_norm,none", "acc_norm_stderr,none"),
    "piqa":          ("acc_norm,none", "acc_norm_stderr,none"),
    "winogrande":    ("acc,none",      "acc_stderr,none"),
    "openbookqa":    ("acc_norm,none", "acc_norm_stderr,none"),
    "mathqa":        ("acc,none",      "acc_stderr,none"),
}
TASK_LABEL = {
    "boolq":         "BoolQ",
    "hellaswag":     "HellaSwag",
    "arc_challenge": "ARC-C",
    "arc_easy":      "ARC-E",
    "piqa":          "PIQA",
    "winogrande":    "WinoGrande",
    "openbookqa":    "OBQA",
    "mathqa":        "MathQA",
}
TASK_COLOR = {
    "boolq":         "#e41a1c",
    "hellaswag":     "#377eb8",
    "arc_challenge": "#4daf4a",
    "arc_easy":      "#ff7f00",
    "piqa":          "#984ea3",
    "winogrande":    "#a65628",
    "openbookqa":    "#f781bf",
    "mathqa":        "#999999",
}

# ── load CSV ──────────────────────────────────────────────────────────────────
def _f(v):
    try: return float(v)
    except: return math.nan

csv_rows = list(csv.DictReader(open(HERE / "llama31_8b.csv")))
for r in csv_rows:
    r["keep_ratio"] = float(r["keep_ratio"])
    r["ppl"]   = _f(r["wikitext2_ppl"])
    r["avg"]   = _f(r["avg_score"])
    r["boolq"] = _f(r["boolq_acc"])

baseline_csv = next(r for r in csv_rows if r["method"] == "baseline")
RATIOS = sorted({r["keep_ratio"] for r in csv_rows if r["method"] != "baseline"})

def csv_series(method, field):
    d = {r["keep_ratio"]: r[field]
         for r in csv_rows if r["method"] == method}
    return [d.get(ratio, math.nan) for ratio in RATIOS]

# ── load JSON (latest timestamp per method+ratio) ────────────────────────────
# filename pattern: Llama-3.1-8B_{method_tag}_{ratio}_{timestamp}.json
# We'll parse method tag by matching against known method directory names.

# Build method tag mapping from filename substrings
METHOD_TAG_MAP = {
    "ASVD_asvd_raw":                        "ASVD_asvd_raw",
    "Basis_sharing_basis_sharing":           "Basis_sharing_basis_sharing",
    "SVDLLMv1_whitening_only":              "SVDLLMv1_whitening_only",
    "SVDLLMv1_whitening_then_update":       "SVDLLMv1_whitening_then_update",
    "SVDLLMv2_v2":                          "SVDLLMv2_v2",
    "DobiSVD_cuij_Llama_3_1_8B_DobiSVD":   "DobiSVD_cuij_Llama_3_1_8B_DobiSVD",
}

def _load_json_data():
    """
    Returns dict: {method: {ratio: {task: (acc, stderr)}}}
    Uses the latest-timestamp file for each (method, ratio).
    """
    data = defaultdict(lambda: defaultdict(dict))
    # group files by method+ratio, pick latest
    groups = defaultdict(list)  # (method, ratio) -> [path]
    for f in JDIR.glob("Llama-3.1-8B_*.json"):
        name = f.stem  # Llama-3.1-8B_<method_tag>_<ratio>_<ts>
        for method, tag in METHOD_TAG_MAP.items():
            if f"_{tag}_" in name:
                # extract ratio
                m = re.search(rf"_{re.escape(tag)}_(\d+\.\d+)_", name)
                if m:
                    ratio = float(m.group(1))
                    groups[(method, ratio)].append(f)
                break
    # baseline
    baseline_files = sorted(JDIR.glob("Llama-3.1-8B_baseline_*.json"))
    baseline_json = {}
    if baseline_files:
        j = json.load(open(baseline_files[-1]))
        for task in TASKS:
            if task in j:
                ak, sk = TASK_KEY[task]
                baseline_json[task] = (j[task].get(ak, math.nan),
                                       j[task].get(sk, math.nan))

    for (method, ratio), files in groups.items():
        latest = sorted(files)[-1]   # lexicographic timestamp sort
        j = json.load(open(latest))
        for task in TASKS:
            if task in j:
                ak, sk = TASK_KEY[task]
                data[method][ratio][task] = (
                    j[task].get(ak, math.nan),
                    j[task].get(sk, math.nan),
                )
    return data, baseline_json

json_data, baseline_json = _load_json_data()

# ── save helper ───────────────────────────────────────────────────────────────
def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {name}")

def _left_title(fig, text, x=0.01):
    fig.text(x, 0.5, text, rotation=90, va="center", ha="center",
             fontsize=10, fontweight="bold")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — PPL vs keep_ratio  (legend at bottom)
# ─────────────────────────────────────────────────────────────────────────────
PPL_CAP = 150_000
fig, ax = plt.subplots(figsize=(7, 5.5))

for method in METHODS_ORDER:
    ppls = [min(p, PPL_CAP) for p in csv_series(method, "ppl")]
    is_false = method in FALSE_PPL_METHODS
    ax.semilogy(
        RATIOS, ppls,
        color=COLOR[method], marker=MARKER[method],
        linestyle="--" if is_false else "-",
        linewidth=1.6, markersize=6,
        label=LABEL_SHORT[method] + (" †" if is_false else ""),
    )

ax.axhline(baseline_csv["ppl"], color="black", linestyle=":", linewidth=1.2,
           label=f"Baseline ({baseline_csv['ppl']:.2f})")

ax.set_xlabel("Keep Ratio")
ax.set_ylabel("WikiText-2 PPL (log scale)")
ax.set_title("WikiText-2 PPL vs Compression Ratio  (Llama-3.1-8B, bf16)", loc="left")
ax.set_xticks(RATIOS)
ax.set_ylim(5, PPL_CAP * 3)

# legend below x-axis
handles, lbls = ax.get_legend_handles_labels()
fig.legend(handles, lbls,
           loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02),
           framealpha=0.9, fontsize=8)

fig.text(0.01, -0.10,
         "† PPL is a false signal: model collapsed; local update optimises "
         "fixed logits toward token-frequency → PPL ≈ unigram entropy.",
         fontsize=7.5, color="#555555", va="top")

fig.tight_layout(rect=[0, 0.13, 1, 1])
save(fig, "fig1_ppl")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Per-method per-task accuracy (2×3 grid, error bands from JSON)
# ─────────────────────────────────────────────────────────────────────────────
SHOW_TASKS = ["boolq", "hellaswag", "arc_challenge", "piqa", "winogrande", "openbookqa"]

fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharey=False)
axes = axes.flatten()

for ax_i, method in enumerate(METHODS_ORDER):
    ax = axes[ax_i]
    is_false = method in FALSE_PPL_METHODS

    for task in SHOW_TASKS:
        accs, errs = [], []
        for ratio in RATIOS:
            entry = json_data[method].get(ratio, {}).get(task)
            if entry:
                accs.append(entry[0])
                errs.append(entry[1])
            else:
                accs.append(math.nan)
                errs.append(math.nan)

        accs = np.array(accs)
        errs = np.array(errs)
        color = TASK_COLOR[task]

        ax.plot(RATIOS, accs, color=color, marker="o", markersize=4,
                linewidth=1.4, label=TASK_LABEL[task])
        # error band
        ax.fill_between(RATIOS, accs - errs, accs + errs,
                        color=color, alpha=0.12)

    # baseline horizontal dashed lines per task
    for task in SHOW_TASKS:
        entry = baseline_json.get(task)
        if entry:
            ax.axhline(entry[0], color=TASK_COLOR[task], linestyle=":",
                       linewidth=0.8, alpha=0.5)

    # degenerate band
    ax.axhspan(0.19, 0.40, color="#dddddd", alpha=0.3, zorder=0)

    title = LABEL[method].replace("\n", " ")
    if is_false:
        title += "  †"
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold")
    ax.set_xticks(RATIOS)
    ax.set_xlabel("Keep Ratio", fontsize=7.5)
    ax.set_ylabel("Accuracy", fontsize=7.5)
    ax.set_ylim(0.15, 0.85)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

# shared legend at bottom
handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls,
           loc="lower center", ncol=len(SHOW_TASKS),
           bbox_to_anchor=(0.5, -0.04), framealpha=0.9, fontsize=8)

_left_title(fig, "Per-Task Accuracy vs Compression Ratio  (Llama-3.1-8B)")

fig.text(0.5, -0.01,
         "Shaded band = ±1 stderr from JSON.  "
         "Grey region = degenerate zone (model outputs constant logits).  "
         "Dotted lines = baseline accuracy per task.  "
         "† PPL false signal.",
         ha="center", fontsize=7.5, color="#555555")

fig.subplots_adjust(left=0.07, hspace=0.42, wspace=0.30)
save(fig, "fig2_per_method")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Avg zero-shot accuracy  (one panel per method, small-multiples)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharey=True)
axes = axes.flatten()

BASELINE_AVG = baseline_csv["avg"]
_DEG_LO, _DEG_HI = 0.31, 0.35

for ax_i, method in enumerate(METHODS_ORDER):
    ax = axes[ax_i]
    is_false = method in FALSE_PPL_METHODS
    avgs  = csv_series(method, "avg")
    boolqs = csv_series(method, "boolq")

    # degenerate band
    ax.axhspan(_DEG_LO, _DEG_HI, color="#dddddd", alpha=0.45, zorder=0,
               label="Degenerate band")
    # baseline
    ax.axhline(BASELINE_AVG, color="black", linestyle=":", linewidth=1.0,
               label=f"Baseline ({BASELINE_AVG:.3f})")

    # colour each point: grey if degenerate boolq, else method colour
    _deg_thresh = 0.40
    point_colors = [
        "#bbbbbb" if b < _deg_thresh else COLOR[method]
        for b in boolqs
    ]
    ax.plot(RATIOS, avgs,
            color=COLOR[method],
            linestyle="--" if is_false else "-",
            linewidth=1.4, zorder=2)
    for x, y, c in zip(RATIOS, avgs, point_colors):
        ax.scatter(x, y, color=c, s=30, zorder=3,
                   marker=MARKER[method])

    title = LABEL[method].replace("\n", " ")
    if is_false:
        title += "  †"
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold")
    ax.set_xticks(RATIOS)
    ax.set_xlabel("Keep Ratio", fontsize=7.5)
    ax.set_ylabel("Avg Accuracy", fontsize=7.5)
    ax.set_ylim(0.28, 0.70)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

# shared legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
leg_handles = [
    Patch(color="#dddddd", alpha=0.7, label="Degenerate band"),
    Line2D([0], [0], color="black", linestyle=":", linewidth=1.0,
           label=f"Baseline ({BASELINE_AVG:.3f})"),
    Line2D([0], [0], color="#bbbbbb", marker="o", linestyle="none",
           markersize=6, label="Degenerate point"),
]
fig.legend(handles=leg_handles,
           loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.06), framealpha=0.9, fontsize=8)

_left_title(fig, "Avg Zero-Shot Accuracy vs Compression Ratio  (Llama-3.1-8B)")

fig.text(0.5, -0.02,
         "Grey points = boolq < 0.40 (degenerate).  "
         "† PPL is a false signal (unigram artifact).",
         ha="center", fontsize=7.5, color="#555555")

fig.subplots_adjust(left=0.07, hspace=0.42, wspace=0.18)
save(fig, "fig3_avg_score")

# ─────────────────────────────────────────────────────────────────────────────
# Table 1 — full LaTeX results
# ─────────────────────────────────────────────────────────────────────────────
TASK_COLS = ["ppl", "boolq", "arc_c", "arc_e", "hella", "wino", "obqa", "piqa", "mathqa", "avg"]
TASK_HDRS = ["PPL↓", "BoolQ", "ARC-C", "ARC-E", "Hella", "Wino", "OBQA", "PIQA", "MathQA", "Avg↑"]

def fmt(val, field):
    if math.isnan(val): return "--"
    if field == "ppl":
        return f"{val/1000:.1f}k" if val >= 1000 else f"{val:.1f}"
    return f"{val:.3f}"

def get_field(r, field):
    mapping = {
        "ppl": "wikitext2_ppl", "boolq": "boolq_acc",
        "arc_c": "arc_challenge_acc_norm", "arc_e": "arc_easy_acc_norm",
        "hella": "hellaswag_acc_norm",    "wino": "winogrande_acc",
        "obqa":  "openbookqa_acc_norm",   "piqa": "piqa_acc_norm",
        "mathqa": "mathqa_acc",           "avg": "avg_score",
    }
    return _f(r.get(mapping.get(field, field), "nan"))

def is_deg(r): return _f(r.get("boolq_acc", "nan")) < 0.40

LABEL_TEX = {
    "ASVD_asvd_raw":                        "ASVD",
    "Basis_sharing_basis_sharing":           "Basis Sharing",
    "SVDLLMv1_whitening_only":              r"SVD-LLM v1 (whitening)",
    "SVDLLMv1_whitening_then_update":       r"SVD-LLM v1 (w.+update)$^{\ddagger}$",
    "SVDLLMv2_v2":                          r"SVD-LLM v2",
    "DobiSVD_cuij_Llama_3_1_8B_DobiSVD":   r"DobiSVD",
}

ncols = 2 + len(TASK_COLS)
lines = [
    r"\begin{table}[t]",
    r"\centering\small",
    r"\caption{Llama-3.1-8B compression benchmark results (bf16, WikiText-2 PPL + 8 zero-shot tasks). "
    r"$^\dagger$~model outputs constant logits (boolq\,$\approx$\,0.378 = label-frequency baseline). "
    r"$^\ddagger$~PPL is a unigram artifact (model collapsed).}",
    r"\label{tab:llama31_8b}",
    r"\begin{tabular}{ll" + "r" * len(TASK_COLS) + r"}",
    r"\toprule",
    r"Method & Ratio & " + " & ".join(TASK_HDRS) + r" \\",
    r"\midrule",
]

# baseline
b = next(r for r in csv_rows if r["method"] == "baseline")
bcells = [r"\textbf{Baseline}", r"\textbf{1.0}"]
for f in TASK_COLS:
    bcells.append(r"\textbf{" + fmt(get_field(b, f), f) + r"}")
lines.append(" & ".join(bcells) + r" \\")
lines.append(r"\midrule")

for mi, method in enumerate(METHODS_ORDER):
    mrows = sorted([r for r in csv_rows if r["method"] == method],
                   key=lambda r: r["keep_ratio"])
    tex_label = LABEL_TEX[method]
    for ji, r in enumerate(mrows):
        cells = []
        cells.append(
            r"\multirow{" + str(len(mrows)) + r"}{*}{" + tex_label + r"}"
            if ji == 0 else ""
        )
        ratio_s = f"{r['keep_ratio']:.1f}"
        if is_deg(r):
            ratio_s += r"$^{\dagger}$"
        cells.append(ratio_s)
        for f in TASK_COLS:
            cells.append(fmt(get_field(r, f), f))
        lines.append(" & ".join(cells) + r" \\")
    if mi < len(METHODS_ORDER) - 1:
        lines.append(r"\cmidrule{1-" + str(ncols) + r"}")

lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUT / "table1_results.tex").write_text("\n".join(lines))
print("✓ table1_results")
print(f"\nAll outputs → {OUT}")
