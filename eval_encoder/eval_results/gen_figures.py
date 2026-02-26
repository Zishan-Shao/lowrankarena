#!/usr/bin/env python3
"""
Generate benchmark figures for draft.md.
Run from repo root:  python eval_encoder/eval_results/gen_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9.5,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Palette
C = {
    'dense':  '#4878CF',   # blue
    'einsum': '#E87B3E',   # orange
    'sdpa':   '#6ACC65',   # green
    'flash':  '#D65F5F',   # red
    'svd':    '#4878CF',
    'fwsvd':  '#E87B3E',
    'drone':  '#6ACC65',
    'adasvd': '#956CB4',
}

# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — Peak Memory across kernel tiers
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.4))

labels = ['Dense\n(HF BERT)', 'Naive\n(einsum)', 'Naive\n(SDPA)', 'FlashSVD']
values = [987, 2004, 1566, 708]
colors = [C['dense'], C['einsum'], C['sdpa'], C['flash']]

bars = ax.bar(labels, values, color=colors, width=0.52,
              edgecolor='white', linewidth=1.5, zorder=3)
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

# Bar top labels: just the value
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 28,
            f'{val} MB', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Vertical transition arrows at midpoints between bars (x = 0.5, 1.5, 2.5)
# % label placed at y_mid, to the right of the arrow
transitions = [
    (0.5, 987,  2004, '+117%', 'logits+A materialize', C['einsum']),
    (1.5, 2004, 1566, '−22%',  'SDPA: no materialize', C['sdpa']),
    (2.5, 1566,  708, '−55%',  'Triton fused kernel',  C['flash']),
]

for x_mid, y_from, y_to, pct, note, color in transitions:
    ax.annotate('', xy=(x_mid, y_to), xytext=(x_mid, y_from),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.2,
                                mutation_scale=15))
    y_mid = (y_from + y_to) / 2
    ax.text(x_mid + 0.06, y_mid,
            f'{pct}\n{note}', va='center', ha='left',
            fontsize=8.5, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.9))


ax.axhline(987, color=C['dense'], linestyle=':', alpha=0.5, linewidth=1.2)
ax.set_ylabel('Peak Inference Memory (MB)')
ax.set_title('Peak Inference Memory under Different Attention Implementations\n'
             '(per-head ra48, seq=512, bs=32, fp32, SVD method)')
ax.set_ylim(0, 2850)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig1_memory_kernels.png'), bbox_inches='tight')
plt.close()
print('✓ fig1_memory_kernels.png')

# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Throughput across kernel tiers
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.8))

# avg across SVD/FWSVD/DRONE/AdaSVD for compressed tiers
labels = ['Dense\n(HF BERT)', 'Naive\n(einsum)', 'Naive\n(SDPA)', 'FlashSVD']
values = [265.7, 193.8, 342.8, 329.3]
colors = [C['dense'], C['einsum'], C['sdpa'], C['flash']]

bars = ax.bar(labels, values, color=colors, width=0.52,
              edgecolor='white', linewidth=1.5, zorder=3)
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
            f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Annotations
ax.annotate('+77%\nvs einsum', xy=(2, 310), xytext=(1.5, 215),
            ha='center', fontsize=10, fontweight='bold', color='#1a1a1a',
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#888888', alpha=0.95))
ax.annotate('−4% vs SDPA', xy=(2.5, 336), ha='center', fontsize=10,
            fontweight='bold', color=C['flash'])

ax.text(0.98, 0.97,
        'FlashSVD: −5% throughput vs SDPA\n−55% memory vs SDPA (see Figure 1)',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                  edgecolor='#AAAAAA', alpha=0.9))

ax.set_ylabel('Throughput (samples / sec)')
ax.set_title('Throughput — Attention Kernel Tiers\n'
             '(per-head ra48, seq=512, bs=32, fp32, avg across SVD/FWSVD/DRONE/AdaSVD)')
ax.set_ylim(0, 430)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig2_throughput_kernels.png'), bbox_inches='tight')
plt.close()
print('✓ fig2_throughput_kernels.png')

# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — G-AVG: Per-head vs Full-matrix  (2-panel: Stage1 + Stage2)
# ──────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

methods = ['SVD', 'FWSVD', 'DRONE', 'AdaSVD']
x = np.arange(len(methods))
w = 0.35

# Panel (a): Stage1
ph_s1 = [0.333, 0.543, 0.594, 0.389]
fm_s1 = [0.421, 0.574, 0.604, 0.400]

b1 = ax1.bar(x - w/2, ph_s1, w, label='Per-head (ra48)',    color='#4878CF', alpha=0.88)
b2 = ax1.bar(x + w/2, fm_s1, w, label='Full-matrix (ra312)', color='#E87B3E', alpha=0.88)

for bar in list(b1) + list(b2):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.009,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8.5)

ax1.axhline(0.827, color='gray', linestyle='--', linewidth=1.4, label='Dense (0.827)')
ax1.set_xticks(x); ax1.set_xticklabels(methods)
ax1.set_ylabel('GLUE Average (G-AVG)')
ax1.set_title('(a) Stage 1 — No Finetune')
ax1.set_ylim(0, 1.0)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.35)

# Panel (b): Stage2
ph_s2 = [0.7737, 0.7963, 0.8020, 0.7837]
# FM stage2 incomplete — only SVD available
fm_s2 = [0.774, None, None, None]   # SVD-FM partial (4 tasks only, exact G-AVG TBD)

b3 = ax2.bar(x - w/2, ph_s2, w, label='Per-head (ra48)',     color='#4878CF', alpha=0.88)
for bar in b3:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.009,
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8.5)

# FM bars only where available
for i, val in enumerate(fm_s2):
    if val is not None:
        b = ax2.bar(x[i] + w/2, val, w, color='#E87B3E', alpha=0.88,
                    label='Full-matrix (ra312)')
        ax2.text(b[0].get_x() + b[0].get_width()/2, b[0].get_height() + 0.009,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=8.5)
    else:
        ax2.text(x[i] + w/2, 0.05, 'pending', ha='center', va='bottom',
                 fontsize=8, color='#E87B3E', rotation=90, alpha=0.6)

ax2.axhline(0.833, color='gray', linestyle='--', linewidth=1.4, label='Dense-ft (0.833)')
ax2.set_xticks(x); ax2.set_xticklabels(methods)
ax2.set_ylabel('GLUE Average (G-AVG)')
ax2.set_title('(b) Stage 2 — Post-compress Finetune')
ax2.set_ylim(0, 1.0)
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.35)

fig.suptitle('GLUE Average: Per-head vs Full-matrix Compression\n'
             '(param_ratio ≈ 0.527, seq=512, bs=32, fp32)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig3_glue_avg_ph_vs_fm.png'), bbox_inches='tight')
plt.close()
print('✓ fig3_glue_avg_ph_vs_fm.png')

# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — MRPC F1: structural collapse & recovery
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 5.2))

x_labels = ['Dense', 'Per-head\n(Stage1)', 'Full-matrix\n(Stage1)',
            'Per-head\n(Stage2-ft)', 'Full-matrix\n(Stage2-ft)']
xpos = np.arange(len(x_labels))

data = {
    'SVD':    [0.913, 0.000, 0.365, 0.843, 0.884],
    'FWSVD':  [0.913, 0.372, 0.803, 0.886, None],
    'DRONE':  [0.913, 0.847, 0.834, 0.902, None],
    'AdaSVD': [0.913, 0.000, 0.129, 0.885, None],
}
m_colors  = {'SVD': C['svd'], 'FWSVD': C['fwsvd'], 'DRONE': C['drone'], 'AdaSVD': C['adasvd']}
m_markers = {'SVD': 'o', 'FWSVD': 's', 'DRONE': '^', 'AdaSVD': 'D'}

ax.axhspan(0, 0.05, alpha=0.12, color='red')
ax.axhline(0.05, color='red', linewidth=0.8, linestyle='--', alpha=0.6)

for method, vals in data.items():
    valid_x = [xpos[i] for i, v in enumerate(vals) if v is not None]
    valid_y = [v for v in vals if v is not None]
    ax.plot(valid_x, valid_y,
            color=m_colors[method], marker=m_markers[method],
            linewidth=2.2, markersize=9, label=method, zorder=4)
    # Mark missing FM-ft with dashed continuation
    for i, v in enumerate(vals):
        if v is None:
            prev = vals[i-1]
            ax.plot([xpos[i-1], xpos[i]], [prev, prev],
                    color=m_colors[method], linewidth=1.2, linestyle=':', alpha=0.45)
            ax.text(xpos[i], prev + 0.015, '(no data)',
                    ha='center', fontsize=7.5, color=m_colors[method], alpha=0.6)

# Annotation: per-head collapse
ax.annotate('Per-head SVD/AdaSVD\ncollapse: all-negative\nF1 = 0',
            xy=(1, 0.0), xytext=(1.15, 0.22),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.3),
            fontsize=9, color='red')

# Annotation: full-matrix recovery
ax.annotate('Full-matrix\npartially recovers',
            xy=(2, 0.365), xytext=(2.3, 0.20),
            arrowprops=dict(arrowstyle='->', color=C['svd'], lw=1.3),
            fontsize=9, color=C['svd'])

ax.set_xticks(xpos)
ax.set_xticklabels(x_labels)
ax.set_ylabel('MRPC F1')
ax.set_title('MRPC F1 — Structural Collapse and Recovery\n'
             '(per-head 768→48/head bottleneck destroys multi-head cooperation;\n'
             'full-matrix global subspace partially recovers; finetune recovers all)')
ax.set_ylim(-0.05, 1.02)
ax.legend(loc='lower right')
ax.grid(axis='y', linestyle='--', alpha=0.35)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_mrpc_collapse.png'), bbox_inches='tight')
plt.close()
print('✓ fig4_mrpc_collapse.png')

# ──────────────────────────────────────────────────────────────────────────────
# Figure 6 — Pareto Front: Memory vs G-AVG (Stage1)
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.8))

# (memory, g_avg, label, method, backend)
points = [
    (987,  0.827, 'Dense',          'Dense',   'dense'),
    (2004, 0.333, 'SVD (Naive)',    'SVD',     'naive'),
    (708,  0.333, 'SVD (Flash)',    'SVD',     'flash'),
    (2011, 0.543, 'FWSVD (Naive)', 'FWSVD',   'naive'),
    (708,  0.543, 'FWSVD (Flash)', 'FWSVD',   'flash'),
    (2004, 0.594, 'DRONE (Naive)', 'DRONE',   'naive'),
    (708,  0.594, 'DRONE (Flash)', 'DRONE',   'flash'),
    (2519, 0.389, 'AdaSVD (Naive)','AdaSVD',  'naive'),
    (723,  0.389, 'AdaSVD (Flash)','AdaSVD',  'flash'),
]

m_colors  = {'Dense': 'black', 'SVD': C['svd'], 'FWSVD': C['fwsvd'],
             'DRONE': C['drone'], 'AdaSVD': C['adasvd']}
m_markers = {'Dense': '*', 'SVD': 'o', 'FWSVD': 's', 'DRONE': '^', 'AdaSVD': 'D'}

# Arrows: Naive → Flash (same method, same accuracy, less memory)
for method in ['SVD', 'FWSVD', 'DRONE', 'AdaSVD']:
    n = next(p for p in points if p[2].startswith(method) and p[4] == 'naive')
    f = next(p for p in points if p[2].startswith(method) and p[4] == 'flash')
    ax.annotate('', xy=(f[0], f[1]), xytext=(n[0], n[1]),
                arrowprops=dict(arrowstyle='->', color=m_colors[method],
                                lw=1.8, alpha=0.55))

# Scatter points
label_offsets = {
    'Dense':          (30,  0.012),
    'SVD (Naive)':    (30,  0.012),
    'SVD (Flash)':    (30, -0.030),
    'FWSVD (Naive)':  (30,  0.012),
    'FWSVD (Flash)':  (30, -0.030),
    'DRONE (Naive)':  (30,  0.012),
    'DRONE (Flash)':  (30, -0.030),
    'AdaSVD (Naive)': (30,  0.012),
    'AdaSVD (Flash)': (30, -0.030),
}

for mem, acc, label, method, backend in points:
    is_flash = backend in ('flash', 'dense')
    ax.scatter(mem, acc,
               c=m_colors[method], marker=m_markers[method],
               s=200 if is_flash else 90,
               alpha=1.0 if is_flash else 0.45,
               edgecolors='white' if is_flash else m_colors[method],
               linewidths=1.8, zorder=5)
    dx, dy = label_offsets[label]
    ax.text(mem + dx, acc + dy, label, fontsize=8.5,
            color=m_colors[method], va='bottom')

# Legend
handles = [mpatches.Patch(color=m_colors[m], label=m)
           for m in ['SVD', 'FWSVD', 'DRONE', 'AdaSVD', 'Dense']]
handles += [
    plt.scatter([], [], marker='o', c='gray', s=90,  alpha=0.45, label='Naive backend'),
    plt.scatter([], [], marker='o', c='gray', s=200, alpha=1.0,  label='FlashSVD backend'),
]
ax.legend(handles=handles, loc='upper right', fontsize=9)

ax.set_xlabel('Peak Inference Memory (MB)')
ax.set_ylabel('GLUE Average G-AVG (Stage1, 8 tasks)')
ax.set_title('Memory–Accuracy Pareto Front (Stage1, No Finetune)\n'
             'Arrows: Naive → FlashSVD  (same accuracy, −55% memory)')
ax.set_xlim(600, 2650)
ax.set_ylim(0.22, 0.93)
ax.grid(linestyle='--', alpha=0.35)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig6_pareto_front.png'), bbox_inches='tight')
plt.close()
print('✓ fig6_pareto_front.png')

# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 — Parameter vs Activation Memory Breakdown (stacked bar)
#
# Parameter memory  = total_compressed_params × 4 bytes (fp32)
#   Dense:   109,483,778 × 4 = 418 MB
#   SVD any:  69,375,746 × 4 = 265 MB  (from encoder_runs.csv total_compressed_params)
#
# Activation memory = peak_mem − param_mem
#   Dense           :  987 − 418 = 569 MB  (SDPA, no [B,H,M,M])
#   SVD Naive einsum: 2004 − 265 = 1739 MB (logits+A each ~384 MB)
#   SVD Naive SDPA  : 1566 − 265 = 1301 MB
#   SVD FlashSVD    :  708 − 265 =  443 MB
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.2))

xlabels = ['Dense\n(HF BERT)', 'SVD\nNaive (einsum)', 'SVD\nNaive (SDPA)', 'SVD\nFlashSVD']
param_mem  = [418, 265, 265, 265]
activ_mem  = [569, 1739, 1301, 443]
colors_p   = [C['dense'], C['einsum'], C['sdpa'], C['flash']]
color_act  = '#AAAAAA'

x = np.arange(len(xlabels))
w = 0.52

bars_p = ax.bar(x, param_mem, w, label='Model Parameters',
                color=colors_p, alpha=0.92, edgecolor='white', linewidth=1.2)
bars_a = ax.bar(x, activ_mem, w, bottom=param_mem,
                label='Activations & Buffers',
                color=color_act, alpha=0.65, edgecolor='white', linewidth=1.2,
                hatch='//')

# Total labels on top
totals = [p + a for p, a in zip(param_mem, activ_mem)]
for xi, total in zip(x, totals):
    ax.text(xi, total + 25, f'{total} MB', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

# Sub-labels inside each segment
for xi, p, a in zip(x, param_mem, activ_mem):
    if p >= 60:
        ax.text(xi, p / 2, f'{p} MB\n(params)', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
    if a >= 100:
        ax.text(xi, p + a / 2, f'{a} MB\n(activ.)', ha='center', va='center',
                fontsize=9, color='#333333', fontweight='bold')

# Highlight: param reduction Dense→SVD
ax.annotate('', xy=(1, 265), xytext=(0, 418),
            arrowprops=dict(arrowstyle='->', color='navy', lw=1.5))
ax.text(0.5, 380, '−37%\nparams', ha='center', fontsize=9,
        color='navy', fontweight='bold')

# Highlight: FlashSVD activation vs Dense
ax.annotate('', xy=(3, 265 + 443), xytext=(0, 418 + 569),
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.5,
                            connectionstyle='arc3,rad=0.18'))
ax.text(1.9, 1100, 'FlashSVD total < Dense\n(708 vs 987 MB)', ha='center',
        fontsize=9, color='#555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFDD',
                  edgecolor='#AAAAAA', alpha=0.9))

ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_ylabel('GPU Memory (MB)')
ax.set_title('Memory Breakdown: Model Parameters vs Activations\n'
             '(per-head ra48, seq=512, bs=32, fp32)\n'
             'SVD reduces param memory; FlashSVD eliminates activation overhead')
ax.set_ylim(0, 2400)
ax.legend(loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.35)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig5_memory_breakdown.png'), bbox_inches='tight')
plt.close()
print('✓ fig5_memory_breakdown.png')

print(f'\nAll figures saved to {OUT_DIR}/')
