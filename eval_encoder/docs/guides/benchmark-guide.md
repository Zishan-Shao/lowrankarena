# Encoder SVD Benchmark Guide

**Last Updated:** 2026-02-09

Complete guide for running encoder SVD compression benchmarks on BERT-family models.

---

## 📁 Directory Structure

```
eval_encoder/
├── run_encoder_benchmark.py          # Main benchmark script
├── BENCHMARK_GUIDE.md                # This file
├── FLASHSVD_PERFORMANCE_ANALYSIS.md  # FlashSVD性能分析
├── FLASHSVD_LONGSEQ_RESULTS.md       # FlashSVD长序列测试结果
│
├── scripts/core/                     # Essential test scripts
│   ├── test_adasvd.sh               # AdaSVD comprehensive test
│   ├── test_flashsvd_memory.sh      # FlashSVD memory efficiency test
│   └── check_progress.sh            # Progress monitoring
│
├── eval_results/                     # Raw results (generated)
│   └── consolidated/                # Organized results
│       ├── all_encoder_benchmarks.csv   # All results combined
│       ├── svd_benchmarks.csv           # SVD method only
│       ├── fwsvd_benchmarks.csv         # FWSVD method only
│       └── adasvd_benchmarks.csv        # AdaSVD method only
│
├── adasvd_refactored/               # AdaSVD with FlashSVD support
│   ├── adasvd_wrapper.py            # Main API
│   ├── adaptive_rank_selection.py   # Training logic
│   └── profile_svd.py, profile_flashsvd.py  # Compression modules
│
└── blocks.py                        # SVD block implementations

```

---

## 🚀 Quick Start

### 1. Basic Benchmark (Dense baseline)

```bash
python run_encoder_benchmark.py \
  --method dense \
  --model_id textattack/bert-base-uncased-SST-2 \
  --task sst2 \
  --seq_len 128 \
  --batch_size 32 \
  --dtype fp16
```

### 2. SVD Compression

```bash
python run_encoder_benchmark.py \
  --method svd \
  --rank 512 \
  --backend naive \
  --model_id textattack/bert-base-uncased-SST-2 \
  --task sst2
```

### 3. Fisher-Weighted SVD (FWSVD)

```bash
python run_encoder_benchmark.py \
  --method fwsvd \
  --rank 512 \
  --backend naive \
  --calib_batches 4 \
  --model_id textattack/bert-base-uncased-SST-2 \
  --task sst2
```

### 4. Adaptive Rank SVD (AdaSVD)

```bash
python run_encoder_benchmark.py \
  --method adasvd \
  --budget 0.5 \
  --backend naive \
  --calib_batches 4 \
  --model_id textattack/bert-base-uncased-SST-2 \
  --task sst2
```

---

## 📊 Comprehensive Test Scripts

### AdaSVD Full Test (5 budgets × 2 backends)

```bash
bash scripts/core/test_adasvd.sh
```

**What it does:**
- Tests budgets: 0.3, 0.4, 0.5, 0.6, 0.7
- Tests backends: naive, flashsvd
- Total: 10 runs
- Output: `eval_results/encoder_runs_sst2_adasvd_refactored_5budgets.csv`

**Known Issues:**
- Budget control not working (all budgets → 66% param_ratio)
- High memory usage (~1177 MB vs expected ~400 MB)

### FlashSVD Memory Efficiency Test

```bash
bash scripts/core/test_flashsvd_memory.sh
```

**What it does:**
- Tests 3 scenarios: short seq, long seq, long seq + large batch
- Compares naive vs flashsvd backends
- Demonstrates FlashSVD's memory-efficiency trade-off

**Results:** See `FLASHSVD_LONGSEQ_RESULTS.md`

---

## ⚙️ Backend Selection

### Naive Backend (Recommended Default)

```bash
--backend naive
```

**Pros:**
- ✅ 3-4x faster than FlashSVD
- ✅ Simpler, no Triton dependencies
- ✅ Works for all architectures

**Cons:**
- ❌ Higher memory usage

**Use for:**
- All speed-critical scenarios
- Standard benchmarks
- Production inference

### FlashSVD Backend

```bash
--backend flashsvd
```

**Pros:**
- ✅ 23-66% memory savings (higher with longer sequences)
- ✅ Enables larger batches or longer sequences

**Cons:**
- ❌ 2-3x slower than naive
- ❌ Requires Triton
- ❌ Not compatible with ModernBERT

**Use for:**
- Memory-constrained environments
- When you can tolerate 2-3x slowdown
- Long sequences (seq_len ≥ 512) for better memory savings

**⚠️ Important:** FlashSVD is a **memory optimization**, NOT a speed optimization!

---

## 🎯 Method Comparison

| Method | Rank Selection | Calibration | Accuracy | Speed | Memory | Use Case |
|--------|---------------|-------------|----------|-------|--------|----------|
| **dense** | N/A | No | 100% baseline | Fastest | Highest | Baseline reference |
| **svd** | Fixed uniform | No | ~89% | Fast (~43ms) | Low (~360MB) | Simple compression |
| **fwsvd** | Fixed uniform | Yes | ~92% | Fast (~43ms) | Low (~367MB) | Better accuracy |
| **adasvd** | Adaptive per-op | Yes | ~90% | Slower (~210ms) | High (~1177MB) | Variable ranks |

**Recommendation:** Use **FWSVD** for best accuracy/speed trade-off with naive backend.

---

## 📈 Typical Results (SST-2, BERT-base)

### Speed (latency per batch, lower is better)

```
Dense:   ~15 ms   (baseline)
SVD:     ~43 ms   (naive backend, rank=512)
FWSVD:   ~43 ms   (naive backend, rank=512)
AdaSVD:  ~210 ms  (naive backend, budget=0.5)
```

### Memory (peak usage, lower is better)

```
Dense:   ~2100 MB  (baseline, seq=512, batch=64)
SVD:     ~360 MB   (naive, seq=128, batch=32)
         ~1961 MB  (naive, seq=512, batch=64)
         ~665 MB   (flashsvd, seq=512, batch=64) ← 66% savings!
```

### Accuracy (SST-2 validation set)

```
Dense:   91.7%  (baseline)
SVD:     89.5%  (rank=512)
FWSVD:   92.2%  (rank=512) ← Best!
AdaSVD:  89.5%  (budget=0.3-0.7)
```

---

## 🔧 Command-Line Arguments

### Required Arguments

- `--model_id`: HuggingFace model name
- `--task`: Task name (sst2, mrpc, cola, etc.)

### Method Arguments

- `--method`: Compression method (dense, svd, fwsvd, drone, adasvd)
- `--rank`: Fixed rank for svd/fwsvd (default: 512)
- `--budget`: Target param ratio for adasvd (0.3-0.7)
- `--scope`: Which layers to compress (default: qkv+ffn)

### Backend Arguments

- `--backend`: Backend implementation (naive, flashsvd)
  - Default: naive
  - Use flashsvd only for memory-constrained scenarios

### Data Arguments

- `--seq_len`: Sequence length (default: 128)
- `--batch_size`: Batch size (default: 32)
- `--dtype`: Data type (fp16, fp32; default: fp16)
- `--calib_batches`: Calibration batches for fwsvd/adasvd (default: 4)

### Output Arguments

- `--out_csv`: Output CSV file (default: eval_results/encoder_runs.csv)
- `--notes`: Notes to add to results row

---

## 📝 Results CSV Format

All results are saved to CSV files with the following columns:

| Column | Description |
|--------|-------------|
| `timestamp` | When the test was run |
| `model_id` | Model identifier |
| `task` | Evaluation task |
| `seq_len` | Sequence length |
| `batch_size` | Batch size |
| `method` | Compression method |
| `rank` | Compression rank (for svd/fwsvd) |
| `budget` | Target param ratio (for adasvd) |
| `backend` | Backend implementation |
| `metric_value` | Task accuracy |
| `latency_ms` | Latency per batch (ms) |
| `throughput_sps` | Throughput (samples/second) |
| `peak_mem_mb` | Peak memory usage (MB) |
| `param_ratio` | Compressed / original parameters |

---

## 🐛 Known Issues & Limitations

### 1. AdaSVD Budget Control

**Issue:** All budgets (0.3-0.7) converge to ~66.5% param_ratio

**Status:** Under investigation

**Workaround:** Use fixed ranks with SVD/FWSVD instead

### 2. AdaSVD Memory Usage

**Issue:** AdaSVD uses 3.3x more memory than SVD (1177 MB vs 360 MB)

**Status:** Under investigation - possible model reload issue

**Impact:** FlashSVD memory savings are minimal for AdaSVD (only 6%)

### 3. FlashSVD Slowdown

**Issue:** FlashSVD is 2-3x slower than naive backend

**Status:** **This is by design!** FlashSVD trades speed for memory.

**See:** `FLASHSVD_LONGSEQ_RESULTS.md` for detailed analysis

### 4. ModernBERT + FlashSVD

**Status:** Not yet supported (requires RoPE + GeGLU kernels)

**Workaround:** Use `--backend naive` for ModernBERT

---

## 💡 Tips & Best Practices

1. **Always run dense baseline first** to establish accuracy reference
2. **Use naive backend** unless memory is critically constrained
3. **FWSVD typically gives best accuracy** for the same rank
4. **Longer sequences benefit more from FlashSVD** (59-66% memory savings)
5. **Check consolidated results** in `eval_results/consolidated/` for easy comparison
6. **Use --notes** to document what you're testing

---

## 📚 References

- **FlashSVD Paper:** "FlashSVD: Memory-Efficient Inference with Streaming for Low-Rank Models"
- **Performance Analysis:** `FLASHSVD_PERFORMANCE_ANALYSIS.md`
- **Long Sequence Results:** `FLASHSVD_LONGSEQ_RESULTS.md`
- **Original FWSVD:** `utils/encoder_utils/fwsvd.py`
- **AdaSVD Implementation:** `adasvd_refactored/`

---

## 🆘 Getting Help

If you encounter issues:

1. Check the known issues section above
2. Review the analysis documents (FLASHSVD_*.md)
3. Check logs in `scripts/core/*.log`
4. Verify your environment has required dependencies (transformers, datasets, triton for flashsvd)

---

**Happy Benchmarking! 🚀**
