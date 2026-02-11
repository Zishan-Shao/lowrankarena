# Encoder Low-Rank Compression Benchmark

Unified benchmark pipeline for evaluating SVD-based compression on BERT-family
encoder models.  Three execution modes:

| Mode | `--method` | `--backend` | Description |
|------|-----------|------------|-------------|
| Dense-Naive | `dense` | `naive` | Original model, standard PyTorch |
| LowRank-Naive | `svd` / `fwsvd` / `drone` / `adasvd` | `naive` | Compressed weights, standard GEMM |
| LowRank-FlashSVD | `svd` / `fwsvd` / `drone` / `adasvd` | `flashsvd` | Compressed weights, Triton kernels |

## Supported Architectures

| Architecture | `--method dense` | `--method svd` | `--method fwsvd/drone/adasvd` | `--backend flashsvd` |
|-------------|:---:|:---:|:---:|:---:|
| **BERT** | Y | Y | Y | Y |
| **RoBERTa** | Y | Y | Y | Y |
| **ModernBERT** | Y | Y | N (not yet) | N (different kernels) |

Architecture is auto-detected from the model config (`model_type`).

## Quick Start

All commands run from the **repository root**.

### 1. Dense baseline (BERT, SST-2)

```bash
python eval_encoder/run_encoder_benchmark.py \
  --model_id textattack/bert-base-uncased-SST-2 \
  --method dense --backend naive --task sst2
```

### 2. Plain SVD at r=32 (works for all architectures)

```bash
python eval_encoder/run_encoder_benchmark.py \
  --model_id textattack/bert-base-uncased-SST-2 \
  --method svd --rank 32 --backend naive --task sst2
```

### 3. FWSVD at r=128, naive backend (BERT/RoBERTa only)

```bash
python eval_encoder/run_encoder_benchmark.py \
  --model_id textattack/bert-base-uncased-SST-2 \
  --method fwsvd --rank 128 --backend naive --task sst2
```

### 4. FWSVD at r=128, FlashSVD backend (BERT/RoBERTa only)

```bash
python eval_encoder/run_encoder_benchmark.py \
  --model_id textattack/bert-base-uncased-SST-2 \
  --method fwsvd --rank 128 --backend flashsvd --task sst2
```

> Runs (3) and (4) produce **identical low-rank weights**; only the execution
> backend differs (standard matmul vs. Triton FlashSVD kernels).

### 5. DRONE (data-aware SVD) at r=128

```bash
python eval_encoder/run_encoder_benchmark.py \
  --model_id textattack/bert-base-uncased-SST-2 \
  --method drone --rank 128 --backend naive --task sst2
```

### 6. RoBERTa model

```bash
python eval_encoder/run_encoder_benchmark.py \
  --model_id textattack/roberta-base-SST-2 \
  --method svd --rank 32 --backend naive --task sst2
```

### 7. ModernBERT model

```bash
python eval_encoder/run_encoder_benchmark.py \
  --model_id answerdotai/ModernBERT-base \
  --method svd --rank 32 --backend naive --task sst2
```

## CLI Reference

```
--model_id        HuggingFace model ID or local path  (default: bert-base-uncased)
--task            sst2 | mnli                          (default: sst2)
--seq_len         Max sequence length                  (default: 128)
--batch_size      Batch size                           (default: 32)
--dtype           fp32 | fp16 | bf16                   (default: fp16)
--device          cuda | cpu                           (default: cuda)
--seed            Random seed                          (default: 0)

--method          dense | svd | fwsvd | drone | adasvd (default: dense)
--rank            Target rank (required for svd/fwsvd/drone)
--budget          Budget ratio for adasvd (e.g. 0.6)
--scope           qkv | ffn | qkv+ffn                  (default: qkv+ffn)

--backend         naive | flashsvd                      (default: naive)

--out_csv         Output CSV path                       (default: eval_results/encoder_runs.csv)
--notes           Optional free-text annotation
--warmup_steps    Warmup iterations before timing       (default: 10)
--measure_steps   Timed iterations                      (default: 50)
--calib_batches   Batches for Fisher/covariance calib   (default: 4)
```

## CSV Output Format

Every run **appends** one row to the CSV (default `eval_results/encoder_runs.csv`):

| Column | Example |
|--------|---------|
| `timestamp` | `2026-02-08T14:30:00` |
| `model_id` | `textattack/bert-base-uncased-SST-2` |
| `task` | `sst2` |
| `seq_len` | `128` |
| `batch_size` | `32` |
| `dtype` | `fp16` |
| `method` | `fwsvd` |
| `rank` | `128` |
| `budget` | `` |
| `scope` | `qkv+ffn` |
| `backend` | `naive` |
| `seed` | `0` |
| `metric_name` | `accuracy` |
| `metric_value` | `0.912844` |
| `latency_ms` | `12.34` |
| `throughput_sps` | `2593.5` |
| `peak_mem_mb` | `456.2` |
| `notes` | `` |
| `git_commit` | `886d557` |

## Architecture

```
eval_encoder/
├── __init__.py
├── run_encoder_benchmark.py   # CLI entry point + orchestration
├── blocks.py                  # NaiveSVDBlock + NaiveModernBertSVDBlock (pure PyTorch)
├── flashsvd_backend.py        # enable_flashsvd() + FlashSVDBlock (Triton, BERT/RoBERTa)
├── encoder_benchmark.md       # This file
└── test_encoder_benchmark_smoke.py
```

### Model architecture handling

| Architecture | Layer path | Norm style | Activation | Position encoding |
|---|---|---|---|---|
| BERT | `model.bert.encoder.layer` | Post-norm | GELU | Absolute |
| RoBERTa | `model.roberta.encoder.layer` | Post-norm | GELU | Absolute |
| ModernBERT | `model.model.layers` | Pre-norm | GeGLU | RoPE |

Architecture is auto-detected via `model.config.model_type`.  BERT and RoBERTa
share the same `NaiveSVDBlock` (identical internal structure); ModernBERT uses
`NaiveModernBertSVDBlock` which handles the fused QKV projection, RoPE, GeGLU
FFN, and pre-norm residual pattern.

### Weight identity guarantee

For any non-dense method, the benchmark first constructs `NaiveSVDBlock` layers
(standard matmul forward).  When `--backend flashsvd` is requested, the function
`enable_flashsvd(model)` replaces each block's forward pass with Triton-kernel
execution while **sharing the same parameter tensors** (no copy).  This ensures
that naive and flashsvd runs use byte-identical weights.

### Compression methods

| Method | Source | What it does | Architectures |
|--------|--------|-------------|---------------|
| `svd` | Inline (plain `torch.linalg.svd`) | Basic truncated SVD | All |
| `fwsvd` | `utils/encoder_utils/fwsvd.py` | Fisher-weighted SVD (gradient importance) | BERT, RoBERTa |
| `drone` | Extracted from `src/encoders/BERTWhiten/` | Data-aware SVD using input covariances | BERT, RoBERTa |
| `adasvd` | `src/encoders/BERTAda/adaptive_rank_selection.py` | Learned per-op rank masks under budget | BERT, RoBERTa |

## Dependencies

- `torch`, `transformers`, `datasets`, `evaluate`
- For `--backend flashsvd`: `triton` + CUDA GPU
- For `--method adasvd`: longer runtime (trains a hypernetwork)
- For ModernBERT: `trust_remote_code=True` is used automatically
