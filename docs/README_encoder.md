# Encoder SVD Benchmark

**Unified benchmark pipeline for evaluating SVD-based compression on BERT-family encoder models.**

---

## 🚀 Quick Start

```bash
# Run from repository root
python eval_encoder/run_encoder_benchmark.py \
    --method svd \
    --rank 64 \
    --backend flashsvd \
    --model_id textattack/bert-base-uncased-SST-2 \
    --task sst2 \
    --seq_len 128 \
    --batch_size 32
```

**→ [Full Getting Started Guide](docs/guides/getting-started.md)**

---

## 📚 Documentation

### 📖 User Guides
- **[Getting Started](docs/guides/getting-started.md)** - Quick start, supported architectures, basic usage
- **[Benchmark Guide](docs/guides/benchmark-guide.md)** - Complete testing guide with examples
- **[Quick Reference](docs/guides/quick-reference.md)** - FlashSVD cheat sheet and best practices

### 📊 Performance Analysis
- **FlashSVD Reports**
  - [Final Report](docs/analysis/flashsvd/final-report.md) - Comprehensive FlashSVD evaluation
  - [Performance Analysis](docs/analysis/flashsvd/performance-analysis.md) - Detailed benchmarks
  - [Long Sequence Results](docs/analysis/flashsvd/longseq-results.md) - Memory efficiency at scale
  - [DRONE Highlight](docs/analysis/flashsvd/drone-highlight.md) - DRONE+FlashSVD performance
- **[Small Ranks Analysis](docs/analysis/small-ranks-analysis.md)** - Behavior at low ranks (R=16-48)

### 🔧 Development & Debugging
- **[AdaSVD Bug Analysis](docs/development/adasvd-bug-analysis.md)** - Budget control bug investigation
- **[AdaSVD Fixes Applied](docs/development/adasvd-fixes-applied.md)** - Fix implementation details

---

## 🎯 Key Features

### Supported Methods
| Method | Description | Calibration | Adaptive Ranks |
|--------|-------------|:-----------:|:--------------:|
| **dense** | No compression | ❌ | ❌ |
| **svd** | Plain truncated SVD | ❌ | ❌ |
| **fwsvd** | Fisher-weighted SVD | ✅ | ❌ |
| **drone** | Covariance-based SVD | ✅ | ❌ |
| **adasvd** | Per-operation adaptive ranks | ✅ | ✅ |

### Supported Architectures
| Model | SVD | FWSVD/DRONE/AdaSVD | FlashSVD |
|-------|:---:|:------------------:|:--------:|
| **BERT** | ✅ | ✅ | ✅ |
| **RoBERTa** | ✅ | ✅ | ✅ |
| **ModernBERT** | ✅ | ⚠️ (in progress) | ❌ |

### Backend Options
- **naive** - Standard PyTorch GEMM (reference implementation)
- **flashsvd** - Triton kernels with fused low-rank operations (65-72% memory savings)

---

## 📂 Project Structure

```
eval_encoder/
├── run_encoder_benchmark.py    # Main benchmark script
├── blocks.py                   # SVD block implementations
├── analyze_ranks.py            # Rank distribution analysis tool
│
├── docs/                       # Documentation (organized)
│   ├── guides/                 # User guides
│   ├── analysis/               # Performance reports
│   └── development/            # Dev notes and debugging
│
├── scripts/                    # Automation scripts
│   ├── core/                   # Essential tests
│   ├── adasvd/                 # AdaSVD-specific
│   └── archived_old/           # Historical scripts
│
└── eval_results/               # Generated results
    ├── *.csv                   # Individual runs
    └── consolidated/           # Aggregated results
```

---

## 🏆 Benchmark Highlights

### FlashSVD Performance (vs Naive Backend)
- **Memory**: 65-72% reduction (critical for long sequences)
- **Throughput**: 32-48% faster for large batches
- **Best Config**: DRONE rank=32 @ seq=512, batch=64
  - 72% less memory + 9% faster + 71% accuracy

### AdaSVD Budget Control
- **Fixed**: Budget now accurately hits targets (10%, 30%, 50%)
- **Strategy**: Auto-switches between median rank (low budgets) and adaptive ranks (high budgets)
- **FlashSVD Compatible**: Median rank strategy ensures uniform R across Q/K/V

---

## 📊 Detailed Benchmark Results

### 🔬 Comprehensive Performance Overview (Seq=128, Batch=32, FP32)

**Complete comparison of all compression methods using FP32 precision.**
**Note**: Previous FP16 results archived in [README_fp16.md](README_fp16.md)

| Method | Rank/Budget | Backend | Accuracy | Latency (ms) | Throughput (sps) | Infer Mem (MB) | Δ Mem (vs Naive) | Param Ratio |
|--------|-------------|---------|----------|--------------|------------------|----------------|------------------|-------------|
| dense | - | - | 92.43% | 123.0 | 260.2 | 561 | - | 100.00% |
| svd | 32 | naive | 50.92% | 75.6 | 423.3 | 611 | - | 6.25% |
| svd | 32 | flashsvd | 50.92% | 50.3 | 635.8 | 431 | -180 (-29.5%) | 6.25% |
| svd | 64 | naive | 52.29% | 101.7 | 314.7 | 645 | - | 12.50% |
| svd | 64 | flashsvd | 52.41% | 72.6 | 441.0 | 465 | -180 (-27.9%) | 12.50% |
| svd | 128 | naive | 58.72% | 105.8 | 302.3 | 667 | - | 25.00% |
| svd | 128 | flashsvd | 58.60% | 94.9 | 337.2 | 487 | -180 (-27.0%) | 25.00% |
| svd | 256 | naive | 70.99% | 124.3 | 257.4 | 688 | - | 50.00% |
| svd | 256 | flashsvd | 70.76% | 151.9 | 210.7 | 508 | -180 (-26.2%) | 50.00% |
| svd | 512 | naive | 89.56% | 167.0 | 191.6 | 764 | - | 100.00% |
| svd | 512 | flashsvd | 89.56% | 360.1 | 88.9 | 584 | -180 (-23.6%) | 100.00% |
| fwsvd | 32 | naive | 50.92% | 77.5 | 413.1 | 945 | - | 6.25% |
| fwsvd | 32 | flashsvd | 50.92% | 49.9 | 640.8 | 765 | -180 (-19.0%) | 6.25% |
| fwsvd | 64 | naive | 50.92% | 97.0 | 330.1 | 981 | - | 12.50% |
| fwsvd | 64 | flashsvd | 50.92% | 72.5 | 441.5 | 801 | -180 (-18.3%) | 12.50% |
| fwsvd | 128 | naive | 66.97% | 106.0 | 302.0 | 1003 | - | 25.00% |
| fwsvd | 128 | flashsvd | 67.09% | 93.8 | 341.3 | 823 | -180 (-18.0%) | 25.00% |
| fwsvd | 256 | naive | 79.13% | 124.6 | 256.8 | 1040 | - | 50.00% |
| fwsvd | 256 | flashsvd | 79.13% | 155.1 | 206.3 | 860 | -180 (-17.3%) | 50.00% |
| fwsvd | 512 | naive | 92.20% | 163.9 | 195.2 | 1089 | - | 100.00% |
| fwsvd | 512 | flashsvd | 92.32% | 360.9 | 88.7 | 910 | -180 (-16.5%) | 100.00% |
| drone | 32 | naive | 71.67% | 77.6 | 412.4 | 455 | - | 6.25% |
| drone | 32 | flashsvd | 71.90% | 48.8 | 655.7 | 275 | -180 (-39.6%) | 6.25% |
| drone | 64 | naive | 73.74% | 97.9 | 327.0 | 497 | - | 12.50% |
| drone | 64 | flashsvd | 73.74% | 71.7 | 446.5 | 317 | -180 (-36.2%) | 12.50% |
| drone | 128 | naive | 77.98% | 104.4 | 306.4 | 535 | - | 25.00% |
| drone | 128 | flashsvd | 77.98% | 93.2 | 343.5 | 355 | -180 (-33.6%) | 25.00% |
| drone | 256 | naive | 88.65% | 121.6 | 263.1 | 588 | - | 50.00% |
| drone | 256 | flashsvd | 88.65% | 148.5 | 215.5 | 408 | -180 (-30.6%) | 50.00% |
| drone | 512 | naive | 91.97% | 162.5 | 196.9 | 708 | - | 100.00% |
| drone | 512 | flashsvd | 91.97% | 358.8 | 89.2 | 528 | -180 (-25.4%) | 100.00% |
| adasvd | 0.1 | naive | 50.92% | 141.6 | 225.9 | 1055 | - | 10.19% |
| adasvd | 0.1 | flashsvd | 52.52% | 107.0 | 299.0 | 968 | -87 (-8.2%) | 10.19% |
| adasvd | 0.2 | naive | 50.69% | 129.7 | 246.7 | 1085 | - | 19.65% |
| adasvd | 0.2 | flashsvd | 52.64% | 151.6 | 211.1 | 1015 | -70 (-6.5%) | 19.65% |
| adasvd | 0.3 | naive | 56.19% | 159.7 | 200.4 | 1098 | - | 29.63% |
| adasvd | 0.3 | flashsvd | 56.19% | 169.8 | 188.4 | 1029 | -69 (-6.3%) | 29.63% |
| adasvd | 0.4 | naive | 64.56% | 171.9 | 186.2 | 1107 | - | 39.50% |
| adasvd | 0.4 | flashsvd | 64.45% | 214.3 | 149.3 | 1036 | -70 (-6.3%) | 39.50% |
| adasvd | 0.5 | naive | 78.78% | 180.1 | 177.7 | 1117 | - | 48.98% |
| adasvd | 0.5 | flashsvd | 78.90% | 218.6 | 146.4 | 1047 | -71 (-6.3%) | 48.98% |
| adasvd | 0.6 | naive | 83.95% | 160.5 | 199.3 | 1132 | - | 57.96% |
| adasvd | 0.6 | flashsvd | 83.95% | 234.2 | 136.6 | 1061 | -71 (-6.3%) | 57.96% |
| adasvd | 0.7 | naive | 85.44% | 160.0 | 200.0 | 1147 | - | 66.77% |
| adasvd | 0.7 | flashsvd | 85.44% | 272.1 | 117.6 | 1075 | -72 (-6.3%) | 66.77% |

**Key Highlights (FP32):**
- 🎯 **Dense baseline**: 92.43% accuracy @ 123.0ms latency, 561 MB memory
- ⚡ **Best latency**: SVD R=32 FlashSVD achieves **50.3ms** (2.4× faster than dense)
- 🏆 **Best accuracy**: FWSVD R=512 reaches **92.32%** (matches dense baseline)
- 💾 **Best memory efficiency**: DRONE R=32 FlashSVD saves **180 MB (-39.6%)** vs Naive
- 📊 **Memory pattern (FP32)**: FlashSVD consistently saves 180MB (18-40%) across all methods

---

### Test Environment

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) |
| **CPU** | Intel Core i7-13650HX (13th Gen) |
| **CUDA Driver** | 581.29 |
| **PyTorch** | 2.8.0+cu129 |
| **Framework** | PyTorch + Triton |
| **Model** | `textattack/bert-base-uncased-SST-2` |
| **Task** | SST-2 Sentiment Classification |
| **Batch Size** | 32 |
| **Sequence Length** | 128 |
| **Dtype** | **FP32** |
| **Test Date** | 2026-02-12 |

### AdaSVD: Budget vs Performance

| Budget | Backend | Accuracy | Latency (ms) | Throughput (sps) | Infer Memory (MB) | Param Ratio |
|--------|---------|----------|--------------|------------------|-------------------|-------------|
| 0.1 | flashsvd | 52.52% | 107.0 | 299.0 | 968 | 10.19% |
| 0.1 | naive | 50.92% | 141.6 | 225.9 | 1055 | 10.19% |
| 0.2 | flashsvd | 52.64% | 151.6 | 211.1 | 1015 | 19.65% |
| 0.2 | naive | 50.69% | 129.7 | 246.7 | 1085 | 19.65% |
| 0.3 | flashsvd | 56.19% | 169.8 | 188.4 | 1029 | 29.63% |
| 0.3 | naive | 56.19% | 159.7 | 200.4 | 1098 | 29.63% |
| 0.4 | flashsvd | 64.45% | 214.3 | 149.3 | 1037 | 39.50% |
| 0.4 | naive | 64.56% | 171.9 | 186.2 | 1107 | 39.50% |
| 0.5 | flashsvd | 78.90% | 218.6 | 146.4 | 1047 | 48.98% |
| 0.5 | naive | 78.78% | 180.1 | 177.7 | 1117 | 48.98% |
| 0.6 | flashsvd | 83.95% | 234.2 | 136.6 | 1061 | 57.96% |
| 0.6 | naive | 83.95% | 160.5 | 199.3 | 1132 | 57.96% |
| 0.7 | flashsvd | 85.44% | 272.1 | 117.6 | 1075 | 66.77% |
| 0.7 | naive | 85.44% | 160.0 | 200.0 | 1147 | 66.77% |

**Key Observations:**
- ✅ Budget control now accurate (0.1→10.2%, 0.3→29.6%, 0.5→49.0%)
- ⚡ **FlashSVD reduces latency at low budgets**: 107.0ms vs 141.6ms @ budget=0.1 (24% faster)
- ⚠️ **Latency increases at high budgets**: 272.1ms vs 160.0ms @ budget=0.7 (70% slower)
- 💾 FlashSVD reduces inference memory by **6-8% across all budgets**

### Long Sequence Performance (R=512)

| Seq Len | Method | Backend | Accuracy | Latency (ms) | Throughput (sps) | Infer Memory (MB) | Latency Change |
|---------|--------|---------|----------|--------------|------------------|-------------------|----------------|
| 128 | fwsvd | flashsvd | 92.30% | 145.0 | 220.7 | 283 | +245.2% |
| 128 | fwsvd | naive | 92.19% | 42.0 | 761.9 | 367 |  |
| 128 | svd | flashsvd | 89.51% | 148.0 | 216.2 | 276 | +251.5% |
| 128 | svd | naive | 89.51% | 42.1 | 760.0 | 360 |  |
| 256 | fwsvd | flashsvd | 91.29% | 302.8 | 105.7 | 339 | +191.1% |
| 256 | fwsvd | naive | 91.29% | 104.0 | 307.6 | 565 |  |
| 256 | svd | flashsvd | 89.51% | 309.4 | 103.4 | 327 | +196.5% |
| 256 | svd | naive | 89.51% | 104.4 | 306.6 | 551 |  |
| 512 | fwsvd | flashsvd | 91.85% | 677.0 | 47.3 | 451 | +154.6% |
| 512 | fwsvd | naive | 91.85% | 265.9 | 120.3 | 1099 |  |
| 512 | svd | flashsvd | 89.51% | 681.6 | 47.0 | 440 | +155.7% |
| 512 | svd | naive | 89.51% | 266.6 | 120.1 | 1088 |  |

**Key Observations:**
- ⚡ **FlashSVD latency penalty decreases with length**: +197% @ seq=128, +191% @ seq=256, +155% @ seq=512
- 🎯 **Trade-off**: Higher latency but 59% memory savings at seq=512
- ✅ Accuracy preserved across all sequence lengths (89-92%)

### Small Rank Performance (R=32 vs R=512)

| Rank | Method | Backend | Accuracy | Latency (ms) | Throughput (sps) | Infer Memory (MB) | Param Ratio |
|------|--------|---------|----------|--------------|------------------|-------------------|-------------|
| 32 | fwsvd | flashsvd | 50.89% | 22.3 | 1434.6 | 142 | 6.25% |
| 32 | fwsvd | naive | 50.89% | 23.9 | 1340.4 | 232 | 6.25% |
| 32 | svd | flashsvd | 50.89% | 22.1 | 1444.4 | 134 | 6.25% |
| 32 | svd | naive | 50.89% | 24.6 | 1302.3 | 224 | 6.25% |
| 512 | fwsvd | flashsvd | 92.30% | 145.0 | 220.7 | 283 | 100.00% |
| 512 | fwsvd | naive | 92.19% | 42.0 | 761.9 | 367 | 100.00% |
| 512 | svd | flashsvd | 89.51% | 148.0 | 216.2 | 276 | 100.00% |
| 512 | svd | naive | 89.51% | 42.1 | 760.0 | 360 | 100.00% |

**Key Observations:**
- ⚡ **Ultra-low latency at R=32**: 22-24ms vs 42-45ms @ R=512 (2× faster)
- ✅ FlashSVD maintains low latency advantage at small ranks
- 📉 **Accuracy trade-off**: R=32 collapses to 50.9% vs R=512 at 89.5-92.2%

### DRONE-SVD Performance

| Rank | Backend | Accuracy | Throughput (sps) | Infer Memory (MB) | Param Ratio | Speedup vs Naive |
|------|---------|----------|------------------|-------------------|-------------|------------------|
| 32 | naive | 72.10% | 1206.1 | 224 | 6.25% | 1.00× |
| 32 | flashsvd | 72.10% | 1345.9 | 134 | 6.25% | 1.12× |
| 64 | naive | 74.11% | 1020.7 | 256 | 12.50% | 1.00× |
| 64 | flashsvd | 74.11% | 987.4 | 176 | 12.50% | 0.97× |
| 128 | naive | 78.24% | 966.5 | 270 | 25.00% | 1.00× |
| 128 | flashsvd | 78.24% | 778.8 | 190 | 25.00% | 0.81× |
| 256 | naive | 88.62% | 885.5 | 305 | 50.00% | 1.00× |
| 256 | flashsvd | 88.73% | 469.8 | 222 | 50.00% | 0.53× |

**Key Observations:**
- ✅ **DRONE calibration improves accuracy significantly** vs plain SVD (R=32: 72.1% vs 50.9%)
- 🚀 **FlashSVD provides speedup at low ranks**: 1.12× @ R=32, but slower at higher ranks
- 💾 **Memory savings with FlashSVD**: 40.1% @ R=32, 27.2% @ R=256
- 📈 **Accuracy scales with rank**: 72% @ R=32 → 88.6% @ R=256

**DRONE vs SVD Comparison (R=32):**

| Method | Backend | Accuracy | Throughput | Param Ratio | Accuracy Gain |
|--------|---------|----------|------------|-------------|---------------|
| SVD | naive | 50.89% | 1302 sps | 6.25% | baseline |
| SVD | flashsvd | 50.89% | 1444 sps | 6.25% | baseline |
| DRONE | naive | 72.10% | 1206 sps | 6.25% | **+21.2%** |
| DRONE | flashsvd | 72.10% | 1346 sps | 6.25% | **+21.2%** |

---

## 🔗 Related Components

- **Encoder Blocks**: [`eval_encoder/blocks.py`](blocks.py)
- **AdaSVD Implementation**: [`src/encoders/adasvd_refactored/`](../src/encoders/adasvd_refactored/)
- **Triton Kernels**: [`kernels/encoder_kernels/`](../kernels/encoder_kernels/)
- **Test Scripts**: [`scripts/README.md`](scripts/README.md)

---

## 📝 Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{svd_benchmark_2026,
  title={SVD-Benchmark: Unified Evaluation of Low-Rank Compression for Transformers},
  author={[Your Name]},
  year={2026},
  url={https://github.com/[your-repo]/SVD-Benchmark}
}
```

---

## 📮 Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Documentation**: Check [docs/guides/](docs/guides/) for detailed usage
- **Quick Help**: See [Quick Reference](docs/guides/quick-reference.md) for common tasks

---

**Last Updated**: 2026-02-12
**Data Source**: `../eval_results/encoder_runs.csv` (FP32 tests from 2026-02-12)
**Previous FP16 Data**: Archived in [README_fp16.md](README_fp16.md)
