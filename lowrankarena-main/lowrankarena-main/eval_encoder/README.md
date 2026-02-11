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

**Last Updated**: 2026-02-10
