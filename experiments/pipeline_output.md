# Pipeline Output Reference

Canonical config: `per_head · ra48/rf256/rw208 · budget=0.527 · bf16 · seq=512 · bs=32`

All experiments run from repo root (`lowrankarena/`).

---

## Exp A — Quality (Accuracy)

**Script:** `benchmark/expA.sh`

### Phases

| Phase | Command | What it runs |
|-------|---------|--------------|
| glue | `PHASES=glue bash benchmark/expA.sh` | 8 GLUE tasks × 4 methods, stage1 + stage2 fine-tune |
| superglue | `PHASES=superglue bash benchmark/expA.sh` | BoolQ, RTE, WiC, COPA, CB, HANS, ANLI-R1/R2/R3 stage1 only |
| superglue_finetune | `PHASES=superglue_finetune bash benchmark/expA.sh` | BoolQ + WiC fine-tune (stage2) |

### Outputs

| File | Description |
|------|-------------|
| `experiments/glue/glue_results_{method}_{timestamp}.json` | Per-run GLUE results (initial + final scores per task) |
| `experiments/results/expA.csv` | All-in-one CSV (GLUE + SuperGLUE + robustness rows, appended per run) |
| `experiments/results/glue_summary.csv` | Deduplicated GLUE summary, one row per (method, qkv_mode, seq_len) |
| `experiments/logs/expA_{timestamp}.log` | Full run log |

### Key variables

```bash
METHODS="svd fwsvd drone adasvd"
TASKS_GLUE="cola sst2 mrpc qqp mnli qnli rte stsb"
TASKS_SUPERGLUE="boolq rte_sg wic copa cb hans anli_r1 anli_r2 anli_r3"
TASKS_SUPERGLUE_FINETUNE="boolq wic"
TWO_STAGE=true          # false = skip stage2 fine-tune
RECOMPRESS=false        # true = force recompression even if checkpoint exists
```

### Collect & summarize

```bash
python benchmark/analysis/collect_glue_results.py   # → experiments/results/glue_summary.csv
python benchmark/analysis/collect_expA_results.py   # → experiments/results/expA_summary.csv (all tiers)
```

### Figures

```bash
python benchmark/figures/gen_figures.py             # → fig1-fig6 in experiments/figs/figures/
```

Output files: `fig1_memory_kernels.png`, `fig2_throughput_kernels.png`, `fig3_glue_avg_ph_vs_fm.png`,
`fig3a_glue_avg_stage1.png`, `fig3b_glue_avg_stage2.png`, `fig4_mrpc_collapse.png`,
`fig5_memory_breakdown.png`, `fig6_pareto_front.png`

---

## Exp B — Backend Performance Microbenchmark

**Script:** `benchmark/expB.sh`

Measures latency / throughput / peak_mem / FLOPs for all 4 backends
(naive, sdpa, flashsvd, flashsvd15) at fixed config.

### Key variables

```bash
TASKS="mnli stsb"       # default: all 8 GLUE tasks
METHODS="svd"           # main table: single method (backends compared directly)
BACKENDS="naive sdpa flashsvd flashsvd15"
INPUT_MODES="real synthetic"
REPEAT=3                # independent repeats for variance control
DTYPE=bf16
SEQ_LEN=512
BATCH_SIZE=32
WARMUP=10  MEASURE=50
```

### Outputs

| File | Description |
|------|-------------|
| `experiments/results/expB.csv` | Latency, throughput, peak_mem, FLOPs, rank_pad_pct, seq_pad_pct |
| `experiments/logs/expB_{timestamp}.log` | Full run log |

### Figures

```bash
# Four separate figures (no in-figure tables):
python benchmark/figures/plot_backend_sweep.py \
    --csv experiments/results/expB.csv \
    --tasks mnli stsb --methods svd fwsvd drone adasvd \
    --dtype bf16 --seq_len 512 --outdir experiments/figs/figures

# Two combined figures (cost: latency+memory / perf: throughput+speedup):
python benchmark/figures/plot_combined_figure.py \
    --csv experiments/results/expB.csv \
    --tasks mnli stsb --methods svd fwsvd drone adasvd \
    --dtype bf16 --seq_len 512 --outdir experiments/figs/figures

# FLOPs breakdown (stacked bar, compute quality vs overhead):
python benchmark/figures/plot_flops_breakdown.py \
    --csv experiments/results/expB.csv \
    --task mnli --dtype bf16 --seq_len 512 --outdir experiments/figs/figures
```

Output files:
- `backend_latency_mnli+stsb_bf16_seq512.png`
- `backend_throughput_mnli+stsb_bf16_seq512.png`
- `backend_speedup_mnli+stsb_bf16_seq512.png`
- `backend_memory_mnli+stsb_bf16_seq512.png`
- `combined_cost_mnli+stsb_bf16_seq512.png` — Latency + Memory
- `combined_perf_mnli+stsb_bf16_seq512.png` — Throughput + Speedup
- `flops_breakdown_mnli_bf16_seq512.png`

---

## Exp C — Scaling (seq_len / batch_size)

**Script:** `benchmark/expC.sh`

Two phases sweeping seq_len and batch_size while keeping everything else fixed.
Reuses expA checkpoints; no recompression.

### Key variables

```bash
PHASES="seqlen batch"
TASKS="mnli stsb"
METHODS="svd adasvd"
BACKENDS="naive sdpa flashsvd flashsvd15"
SEQ_LENS="128 256 384 512"    # seqlen phase
BATCH_SIZES="8 16 32 64"      # batch phase
BATCH_FIXED=32                # fixed batch for seqlen sweep
SEQ_FIXED=512                 # fixed seq for batch sweep
DTYPE=bf16
INPUT_MODE=synthetic          # 0% padding for fair backend comparison
REPEAT=3
```

### Outputs

| File | Description |
|------|-------------|
| `experiments/results/expC_seqlen.csv` | Throughput/memory vs seq_len (batch=32 fixed) |
| `experiments/results/expC_batch.csv`  | Throughput/memory vs batch_size (seq=512 fixed) |
| `experiments/logs/expC_{timestamp}.log` | Full run log |

### Figures

```bash
# seq_len scaling (fp32, hardcoded data — do NOT regenerate without new measurements):
python benchmark/figures/plot_seqlen_scaling.py
# → seqlen_memory.{png,pdf}, seqlen_throughput.{png,pdf}, seqlen_reduction.{png,pdf}

# dtype × backend scaling (bf16 from expC_seqlen.csv + fp32 hardcoded):
python benchmark/figures/plot_dtype_scaling.py
# → dtype_memory_scaling.{png,pdf}, dtype_memory_reduction.{png,pdf}, dtype_throughput_scaling.{png,pdf}

# batch scaling:
python benchmark/figures/plot_seqlen_scaling.py  # (batch figures embedded in same script)
# → batch_memory.{png,pdf}, batch_throughput.{png,pdf}
```

> **Note:** `seqlen_*.png/pdf` are archived in `experiments/figs/figures/archive_fp32_*/`.
> Do not overwrite — those numbers are from specific commits (dde6df0 / 66aceb9 / 13b6d39).

---

## Exp D — Kernel-Level Analysis (nsys / ncu)

**Script:** `benchmark/expD.sh`

NVIDIA Nsight Systems + Nsight Compute profiling on 6 representative
(method, backend) pairs for MNLI at canonical config.

### Key variables

```bash
PHASES="nsys ncu"       # nsys = time distribution; ncu = CTA/occupancy root cause
TASK=mnli
DTYPE=bf16
SEQ_LEN=512
BATCH_SIZE=32
GPU_ID=0                # recommended: bind to specific GPU
TAG=mnli_bf16_s512_b32  # auto-derived if not set
```

### 6 profiling points

| Point | Method | Backend |
|-------|--------|---------|
| mnli_svd_naive | SVD | naive |
| mnli_svd_flashsvd | SVD | flashsvd v1.0 |
| mnli_svd_flashsvd15 | SVD | flashsvd v1.5 |
| mnli_adasvd_naive | AdaSVD | naive |
| mnli_adasvd_flashsvd | AdaSVD | flashsvd v1.0 |
| mnli_adasvd_flashsvd15 | AdaSVD | flashsvd v1.5 |

### Outputs

| File | Description |
|------|-------------|
| `experiments/results/expD_mnli_bf16_s512_b32.csv` | Parsed nsys/ncu metrics per point |
| `eval_encoder/eval_results/nsys/nsys_summary.txt` | Raw nsys text summary |
| `eval_encoder/eval_results/nsys/nsys_parsed.csv`  | Parsed nsys kernel stats |
| `experiments/logs/expD_{timestamp}.log` | Full run log |

### Figures

```bash
python benchmark/figures/plot_nsys_kernel.py \
    --csv  experiments/results/expD_mnli_bf16_s512_b32.csv \
    --outdir experiments/figs/figures
# → nsys_kernel_analysis_mnli_bf16_seq512.png
```

---

## Exp E — Supplementary Experiments (E-1 through E-3b)

**Script:** `benchmark/expE.sh`

Four sub-experiments providing supporting evidence for the paper.

### Phases

| Phase | Description | Output |
|-------|-------------|--------|
| E-1 | Timing boundary documentation (print only, no measurements) | — |
| E-2 | Logit alignment check (flashsvd vs naive, numerical equivalence) | `experiments/results/expE_alignment.csv` |
| E-3a | Per-step training time (fwd + bwd + opt), naive/sdpa only | `experiments/results/expE_train_timing.csv` |
| E-3b | Fine-tune recovery curve (accuracy vs training step count) | `experiments/results/expE_recovery.csv` |

### Key variables

```bash
PHASES="e1 e2 e3a e3b"
TASKS="mnli mrpc"
METHODS="svd fwsvd drone"   # no adasvd in E-3 (training only)
E2_TASKS="mnli"
E3A_BACKENDS="naive sdpa"   # flashsvd* → auto-SKIP (no autograd)
E3B_EVAL_STEPS="0 200 500 1000"
E3B_NUM_EPOCHS=3
```

---

## Master Figure Script

Regenerate all non-archived figures in one shot:

```bash
bash benchmark/figures/run_all_figures.sh
# Requires: expA.csv / glue_summary.csv, expB.csv, expC_seqlen.csv, expD CSV
# Output: experiments/figs/figures/
```

---

## Figure Inventory

| Figure | Script | Source CSV | Description |
|--------|--------|-----------|-------------|
| fig1_memory_kernels | gen_figures.py | glue_summary.csv | Kernel memory comparison |
| fig2_throughput_kernels | gen_figures.py | glue_summary.csv | Kernel throughput comparison |
| fig3_glue_avg_ph_vs_fm | gen_figures.py | glue_summary.csv | Per-head vs full-matrix G-AVG |
| fig3a/3b_glue_avg | gen_figures.py | glue_summary.csv | Stage1/2 G-AVG bar charts |
| fig4_mrpc_collapse | gen_figures.py | glue_summary.csv | MRPC collapse analysis |
| fig5_memory_breakdown | gen_figures.py | glue_summary.csv | Memory breakdown |
| fig6_pareto_front | gen_figures.py | glue_summary.csv | Accuracy-memory Pareto |
| backend_latency/throughput/speedup/memory | plot_backend_sweep.py | expB.csv | 4 separate backend figures |
| combined_cost / combined_perf | plot_combined_figure.py | expB.csv | 2×2 combined (cost / perf) |
| flops_breakdown | plot_flops_breakdown.py | expB.csv | FLOPs stacked bar |
| seqlen_memory/throughput/reduction (**archived**) | plot_seqlen_scaling.py | hardcoded | fp32 seq_len scaling |
| dtype_memory_scaling/reduction, dtype_throughput_scaling | plot_dtype_scaling.py | expC_seqlen.csv | dtype × backend scaling |
| batch_memory/throughput | plot_seqlen_scaling.py | hardcoded | batch scaling |
| nsys_kernel_analysis | plot_nsys_kernel.py | expD CSV | Kernel time distribution |
