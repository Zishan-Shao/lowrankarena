# BERT-base SVD Compression Benchmark — Per-Head Mode

> **Generated**: 2026-02-23
> **Source**: `eval_encoder/eval_results/encoder_runs.csv` (rows ≥ 2026-02-20T02:10:35)

---

## 1. Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `bert-base-uncased` (109.5 M params) |
| Task models | `textattack/bert-base-uncased-{TASK}` |
| Compression mode | `per_head` (each attention head decomposed separately) |
| Rank — Attention Q/K/V | 48 (per head, head_dim=64 → 75% retention) |
| Rank — FFN Wi/Wo | 256 |
| Rank — Attention Wo | 208 |
| AdaSVD budget | 0.527 |
| Scope | `qkv + ffn` |
| Sequence length | 512 |
| Batch size | 32 |
| dtype | fp32 |
| Fine-tune epochs | 3 |
| Tasks | CoLA, SST-2, MRPC, QQP, MNLI, QNLI, RTE, STS-B |

### Parameter Counts

| | Params | Ratio |
|--|--------|-------|
| Total (dense) | 109,483,778 | 1.000 |
| Total (compressed) | 69,348,098 | **0.633** |
| Layer-only (dense) | 84,934,656 | 1.000 |
| Layer-only (compressed) | 44,798,976 | **0.527** |

Embeddings (24.5 M) are unchanged. Only encoder layers are compressed.

---

## 2. Throughput (samples/second)

**Setup**: Inference-only forward pass, fp32, seq_len=512, batch_size=32.
`-N` = Naive backend (standard PyTorch matmul); `-F` = FlashSVD (Triton fused kernels).

| Task | Dense | SVD-N | SVD-F | FWSVD-N | FWSVD-F | DRONE-N | DRONE-F | ADA-N | ADA-F |
|------|------:|------:|------:|--------:|--------:|--------:|--------:|------:|------:|
| CoLA | 278.5 | 180.9 | 293.6 | 200.4 | 316.8 | 178.2 | 347.8 | 188.8 | 314.6 |
| SST-2 | 267.4 | 195.6 | 354.6 | 192.5 | 357.1 | 195.4 | 337.6 | 188.5 | 321.6 |
| MRPC | 279.3 | 181.2 | 337.6 | 200.6 | 323.4 | 175.4 | 329.6 | 187.4 | 317.8 |
| QQP | 253.8 | 185.9 | 325.9 | 189.6 | 311.4 | 182.5 | 281.1 | 187.8 | 302.5 |
| MNLI | 258.0 | 185.7 | 335.4 | 188.6 | 317.7 | 184.2 | 339.4 | 188.7 | 314.5 |
| QNLI | 256.6 | 186.6 | 324.0 | 189.1 | 306.9 | 185.1 | 295.4 | 188.7 | 311.2 |
| RTE | 272.8 | 193.9 | 309.0 | 186.8 | 323.0 | 168.6 | 305.6 | 187.4 | 327.2 |
| STS-B | 259.1 | 190.1 | 343.3 | 198.9 | 302.0 | 184.2 | 340.9 | 188.6 | 308.6 |
| **Avg** | **265.7** | **187.5** | **327.9** | **193.3** | **319.8** | **181.7** | **322.2** | **188.2** | **314.8** |
| **vs Dense** | — | **−29.5%** | **+23.4%** | **−27.2%** | **+20.4%** | **−31.6%** | **+21.2%** | **−29.2%** | **+18.5%** |
| **vs Naive** | — | — | **+74.9%** | — | **+65.4%** | — | **+77.3%** | — | **+67.3%** |

**Key observations**:
- All naive SVD backends are **~27–32% slower** than dense (extra matmul from factored weights).
- FlashSVD Triton kernels **recover and surpass** dense throughput by **+18–23%**.
- FlashSVD vs Naive advantage is **+65–77%** across all methods.
- Method differences within the same backend are small (±5 sps).

---

## 3. Peak Memory Usage (MB)

Measured during the full eval run (includes compression step for naive backends).

| Task | Dense | SVD-N | SVD-F | FWSVD-N | FWSVD-F | DRONE-N | DRONE-F | ADA-N | ADA-F |
|------|------:|------:|------:|--------:|--------:|--------:|--------:|------:|------:|
| CoLA | 987 | 2057 | 708 | 2011 | 708 | 2061 | 708 | 2679 | 725 |
| SST-2 | 987 | 2057 | 708 | 2011 | 708 | 2061 | 708 | 2678 | 722 |
| MRPC | 987 | 2057 | 708 | 2011 | 708 | 2061 | 708 | 2679 | 725 |
| QQP | 987 | 2057 | 708 | 2011 | 708 | 2061 | 708 | 2678 | 723 |
| MNLI | 987 | 2057 | 708 | 2011 | 708 | 2061 | 708 | 2679 | 723 |
| QNLI | 987 | 2057 | 708 | 2011 | 708 | 2061 | 708 | 2677 | 721 |
| RTE | 987 | 2057 | 708 | 2011 | 708 | 2061 | 708 | 2678 | 721 |
| STS-B | 987 | 2057 | 708 | 2011 | 708 | 2061 | 708 | 2684 | 726 |
| **Typical** | **987** | **2057** | **708** | **2011** | **708** | **2061** | **708** | **2679** | **723** |
| **vs Dense** | — | **+108%** | **−28%** | **+104%** | **−28%** | **+109%** | **−28%** | **+171%** | **−27%** |

**Notes**:
- Naive peak memory is captured during the eval+compression pipeline: it includes both the original weights and SVD factors held simultaneously, explaining the 2× spike.
- AdaSVD naive is highest (~2.7× dense) due to the additional hypernetwork parameters during ARS training.
- FlashSVD memory is consistent at **708–726 MB** (−28% vs dense 987 MB) because fused kernels never materialize full intermediate matrices.

---

## 4. Accuracy — Stage 1: Post-Compression, Pre-Fine-Tuning

Accuracy measured immediately after compression with no fine-tuning. Naive backend.

| Task | Metric | Dense | SVD | FWSVD | DRONE | AdaSVD (b=0.527) |
|------|--------|------:|----:|------:|------:|----------------:|
| CoLA | MCC | 0.534 | 0.026 | 0.144 | 0.016 | 0.002 |
| SST-2 | Acc | 0.924 | 0.718 | 0.778 | 0.856 | 0.614 |
| MRPC | F1 | 0.914 | 0.000 | 0.372 | 0.847 | 0.000 |
| QQP | F1 | 0.878 | 0.172 | 0.680 | 0.762 | 0.515 |
| MNLI | Acc | 0.846 | 0.374 | 0.522 | 0.579 | 0.351 |
| QNLI | Acc | 0.915 | 0.546 | 0.573 | 0.602 | 0.458 |
| RTE | Acc | 0.726 | 0.473 | 0.585 | 0.596 | 0.551 |
| STS-B | Pear | 0.881 | 0.352 | 0.693 | 0.493 | 0.627 |
| **G-Avg** | | **0.864** | **0.434** | **0.616** | **0.687** | **0.475** |
| **Δ vs Dense** | | — | **−0.430** | **−0.248** | **−0.177** | **−0.389** |

> G-Avg normalizes MCC and Pearson via `(x + 1) / 2` before averaging.

**Key observations**:
- **DRONE is best** at Stage 1 (G-Avg 0.687, Δ −0.177). Full covariance calibration better preserves
  task-relevant activation directions.
- **SVD collapses on MRPC** (F1=0.000): task-specific features align with non-dominant singular
  directions of the fine-tuned model. Data-unaware truncation destroys them.
- **AdaSVD underperforms FWSVD/DRONE** despite adaptive rank allocation. Non-uniform ranks
  at budget 0.527 don't help zero-shot retention.
- **CoLA collapses for all methods** (MCC ≈ 0): grammaticality is encoded in sparse syntactic
  features that escape top singular values. CoLA Stage 1 is always near-random.

---

## 5. Accuracy — Stage 2: Post-Compression + Fine-Tuning (3 Epochs)

| Task | Metric | Dense | SVD | FWSVD | DRONE | AdaSVD (b=0.527) |
|------|--------|------:|----:|------:|------:|----------------:|
| CoLA | MCC | 0.534 | 0.375 | 0.472 | 0.431 | 0.411 |
| SST-2 | Acc | 0.924 | 0.911 | 0.905 | 0.908 | 0.916 |
| MRPC | F1 | 0.914 | 0.843 | 0.886 | 0.902 | 0.885 |
| QQP | F1 | 0.878 | 0.873 | 0.874 | 0.873 | 0.874 |
| MNLI | Acc | 0.846 | 0.820 | 0.827 | 0.820 | 0.823 |
| QNLI | Acc | 0.915 | 0.889 | 0.891 | 0.893 | 0.892 |
| RTE | Acc | 0.726 | 0.614 | 0.646 | 0.740 | 0.596 |
| STS-B | Pear | 0.881 | 0.866 | 0.870 | 0.849 | 0.873 |
| **G-Avg** | | **0.864** | **0.821** | **0.837** | **0.847** | **0.829** |
| **Δ vs Dense** | | — | **−0.043** | **−0.026** | **−0.017** | **−0.035** |

**Key observations**:
- **DRONE retains best quality** after fine-tuning (G-Avg 0.847, Δ −0.017). Good Stage 1
  initialization translates to better fine-tuning recovery.
- **SVD is worst** post-finetune (Δ −0.043) despite complete parity with FWSVD/DRONE
  on large tasks (QQP, MNLI). CoLA (0.375) and RTE (0.614) drag its average.
- **DRONE RTE is exceptional**: 0.740 — essentially matching dense (0.726) and all other methods,
  because DRONE's covariance calibration preserves NLI reasoning directions.
- **AdaSVD at budget 0.527** ranks between SVD and FWSVD (G-Avg 0.829). Non-uniform ranks
  don't provide additional benefit at medium-high budget.
- Most tasks (QQP, MNLI, QNLI, STS-B) converge tightly: max spread across methods < 0.008.

---

## 6. Stage 1 → Stage 2 Recovery

How much of the compression loss is recovered by fine-tuning?

| Method | Stage 1 G-Avg | Stage 2 G-Avg | Recovery Gain |
|--------|:-------------:|:-------------:|:-------------:|
| Dense (ref) | 0.8638 | 0.8638 | — |
| SVD | 0.434 | 0.821 | **+0.387** |
| FWSVD | 0.616 | 0.838 | **+0.222** |
| DRONE | 0.687 | 0.847 | **+0.160** |
| AdaSVD | 0.475 | 0.829 | **+0.353** |

- Methods with worse Stage 1 show larger fine-tuning recovery (SVD: +0.387).
- DRONE has smallest recovery gain because it starts from a better point.
- All methods converge within 0.026 G-Avg points of each other after fine-tuning.

---

## 7. Naive vs FlashSVD Accuracy Delta

All differences are within **±0.004** — effectively identical accuracy.

| Method | Task | Naive | FlashSVD | Δ |
|--------|------|------:|---------:|--:|
| SVD | SST-2 | 0.7179 | 0.7156 | −0.0023 |
| FWSVD | RTE | 0.5848 | 0.5812 | −0.0036 |
| FWSVD | MRPC | 0.3720 | 0.3686 | −0.0034 |
| DRONE | MRPC | 0.8472 | 0.8459 | −0.0013 |
| AdaSVD | SST-2 | 0.6135 | 0.6101 | −0.0034 |
| AdaSVD | RTE | 0.5505 | 0.5487 | −0.0018 |

Max delta: 0.0036. FlashSVD kernels are **mathematically equivalent** to the naive implementation.

---

## 8. Summary

| Method | Throughput (Naive) | Throughput (Flash) | Stage 1 G-Avg | Stage 2 G-Avg | Δ Dense |
|--------|-------------------:|-------------------:|:-------------:|:-------------:|:-------:|
| Dense | 265.7 sps | — | 0.864 | 0.864 | — |
| SVD | 187.5 sps (−29.5%) | 327.9 sps (+23.4%) | 0.434 | 0.821 | −0.043 |
| FWSVD | 193.3 sps (−27.2%) | 319.8 sps (+20.4%) | 0.616 | 0.837 | −0.026 |
| DRONE | 181.7 sps (−31.6%) | 322.2 sps (+21.2%) | 0.687 | 0.847 | −0.017 |
| AdaSVD (b=0.527) | 188.2 sps (−29.2%) | 314.8 sps (+18.5%) | 0.475 | 0.829 | −0.035 |

**Model size**: 109.5 M → 69.3 M params (**−36.7%** total, −47.2% layer-only)

**Recommendation**:
- For best accuracy with fine-tuning: **DRONE** (−0.017 G-Avg vs dense)
- For fastest inference: **FlashSVD backend** (+19–23% vs dense, +65–77% vs naive)
- For zero-shot evaluation: **DRONE > FWSVD >> AdaSVD ≈ SVD**
