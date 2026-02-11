# Encoder SVD Documentation Index

**Last Updated**: 2026-02-11

This directory contains all documentation for encoder SVD implementations.

---

## 📁 Directory Structure

```
docs/
├── README.md                 # Main documentation entry point
├── INDEX.md                  # This file
│
├── modernbert/              # ModernBERT SVD-LLM Implementation
│   ├── EVALUATION_FINETUNED.md
│   ├── EVALUATION_RESULTS.md
│   ├── FINAL_SUMMARY.md
│   ├── RESULTS_COMPARISON.md
│   ├── V1_IMPLEMENTATION_SUMMARY.md
│   └── V2_IMPLEMENTATION_SUMMARY.md
│
├── roberta/                 # RoBERTa Whitening Implementation
│   ├── IMPLEMENTATION_REPORT.md
│   ├── README.md
│   ├── TEACHER_RELEASE_ENHANCEMENT.md
│   └── TEST_RESULTS.md
│
├── adasvd/                  # AdaSVD Implementation
│   └── README.md
│
├── analysis/                # Performance Analysis & Benchmarks
│   ├── flashsvd/
│   │   ├── drone-highlight.md
│   │   ├── final-report.md
│   │   ├── longseq-results.md
│   │   └── performance-analysis.md
│   └── small-ranks-analysis.md
│
├── development/             # Development Notes & Design Docs
│   ├── adasvd-bug-analysis.md
│   ├── adasvd-fixes-applied.md
│   ├── peak-memory-analysis.md
│   ├── performance-measurement-comparison.md
│   ├── svdllm-bert-design.md
│   ├── svdllm-v1-vs-v2-results.md
│   ├── svdllm-v2-update-strategy.md
│   └── v2-data-flow-issue.md
│
├── guides/                  # User Guides & References
│   ├── benchmark-guide.md
│   ├── full-validation-mode.md
│   ├── getting-started.md
│   └── quick-reference.md
│
└── results/                 # Benchmark Results
    └── svdllm-baseline-comparison.md
```

---

## 📚 Documentation by Topic

### 1. ModernBERT SVD-LLM Implementation

**Complete implementation of SVD-LLM v1 & v2 for ModernBERT**

| Document | Description | Status |
|----------|-------------|--------|
| [EVALUATION_FINETUNED.md](modernbert/EVALUATION_FINETUNED.md) | Full evaluation with fine-tuned model (mrm8488/ModernBERT-base-ft-sst2) | ✅ Final |
| [RESULTS_COMPARISON.md](modernbert/RESULTS_COMPARISON.md) | Base vs Fine-tuned comparison (Chinese summary) | ✅ Final |
| [FINAL_SUMMARY.md](modernbert/FINAL_SUMMARY.md) | Complete implementation summary | ✅ Final |
| [EVALUATION_RESULTS.md](modernbert/EVALUATION_RESULTS.md) | Evaluation with base model (answerdotai/ModernBERT-base) | ✅ Reference |
| [V1_IMPLEMENTATION_SUMMARY.md](modernbert/V1_IMPLEMENTATION_SUMMARY.md) | v1 Whitening-SVD implementation details | ✅ Reference |
| [V2_IMPLEMENTATION_SUMMARY.md](modernbert/V2_IMPLEMENTATION_SUMMARY.md) | v2 Local Update implementation details | ✅ Reference |

**Key Findings**:
- ✅ v1 Accuracy (fine-tuned): **85.27%**
- ✅ v2 Accuracy (fine-tuned): **85.83%** (+0.56% improvement)
- ✅ v2 requires fine-tuned teacher (base model drops -6.25%)
- ✅ Conservative strategy (Vo + V2 only) is stable and effective

**Quick Start**: Read [RESULTS_COMPARISON.md](modernbert/RESULTS_COMPARISON.md) for Chinese summary, or [EVALUATION_FINETUNED.md](modernbert/EVALUATION_FINETUNED.md) for detailed English report.

---

### 2. RoBERTa Whitening Implementation

**RoBERTa SVD-LLM v1 & v2 with whitening**

| Document | Description | Status |
|----------|-------------|--------|
| [README.md](roberta/README.md) | Overview and usage guide | ✅ Active |
| [IMPLEMENTATION_REPORT.md](roberta/IMPLEMENTATION_REPORT.md) | Detailed implementation report | ✅ Active |
| [TEST_RESULTS.md](roberta/TEST_RESULTS.md) | Evaluation results and benchmarks | ✅ Active |
| [TEACHER_RELEASE_ENHANCEMENT.md](roberta/TEACHER_RELEASE_ENHANCEMENT.md) | Memory optimization for v2 | ✅ Active |

**Key Features**:
- Post-norm architecture (different from ModernBERT)
- GELU activation (not GeGLU)
- Absolute positional encoding (not RoPE)
- Validated v2 local update strategy

---

### 3. AdaSVD Implementation

**Adaptive rank selection for SVD compression**

| Document | Description | Status |
|----------|-------------|--------|
| [README.md](adasvd/README.md) | Implementation overview and fixes | ✅ Active |

**Key Topics**:
- Budget control fixes (3 critical bugs fixed)
- FlashSVD compatibility (median rank strategy)
- Per-operation adaptive ranks

For detailed analysis, see [development/adasvd-bug-analysis.md](development/adasvd-bug-analysis.md) and [development/adasvd-fixes-applied.md](development/adasvd-fixes-applied.md).

---

## 🔬 Analysis & Benchmarks

### FlashSVD Performance Analysis

| Document | Topic |
|----------|-------|
| [analysis/flashsvd/final-report.md](analysis/flashsvd/final-report.md) | Comprehensive FlashSVD evaluation |
| [analysis/flashsvd/performance-analysis.md](analysis/flashsvd/performance-analysis.md) | Speed and memory analysis |
| [analysis/flashsvd/drone-highlight.md](analysis/flashsvd/drone-highlight.md) | DRONE method highlights |
| [analysis/flashsvd/longseq-results.md](analysis/flashsvd/longseq-results.md) | Long sequence benchmarks |

### Other Analysis

| Document | Topic |
|----------|-------|
| [analysis/small-ranks-analysis.md](analysis/small-ranks-analysis.md) | Small rank configurations |

---

## 🛠️ Development Notes

### Design & Architecture

| Document | Topic |
|----------|-------|
| [development/svdllm-bert-design.md](development/svdllm-bert-design.md) | BERT SVD-LLM design principles |
| [development/svdllm-v2-update-strategy.md](development/svdllm-v2-update-strategy.md) | v2 local update strategy analysis |

### Bug Analysis & Fixes

| Document | Topic |
|----------|-------|
| [development/adasvd-bug-analysis.md](development/adasvd-bug-analysis.md) | AdaSVD budget control bugs |
| [development/adasvd-fixes-applied.md](development/adasvd-fixes-applied.md) | Applied fixes and validation |
| [development/v2-data-flow-issue.md](development/v2-data-flow-issue.md) | v2 data flow debugging |

### Performance Measurement

| Document | Topic |
|----------|-------|
| [development/peak-memory-analysis.md](development/peak-memory-analysis.md) | Memory usage tracking |
| [development/performance-measurement-comparison.md](development/performance-measurement-comparison.md) | Measurement methodology |

### Results Comparison

| Document | Topic |
|----------|-------|
| [development/svdllm-v1-vs-v2-results.md](development/svdllm-v1-vs-v2-results.md) | v1 vs v2 comparison across models |

---

## 📖 User Guides

| Document | Topic | Audience |
|----------|-------|----------|
| [guides/getting-started.md](guides/getting-started.md) | Quick start guide | New users |
| [guides/benchmark-guide.md](guides/benchmark-guide.md) | How to run benchmarks | All users |
| [guides/quick-reference.md](guides/quick-reference.md) | Command reference | All users |
| [guides/full-validation-mode.md](guides/full-validation-mode.md) | Full validation testing | Advanced users |

---

## 📊 Benchmark Results

| Document | Topic |
|----------|-------|
| [results/svdllm-baseline-comparison.md](results/svdllm-baseline-comparison.md) | SVD-LLM baseline comparisons |

---

## 🎯 Quick Navigation

### For New Users
1. Start with [guides/getting-started.md](guides/getting-started.md)
2. Read [guides/benchmark-guide.md](guides/benchmark-guide.md)
3. Check [modernbert/RESULTS_COMPARISON.md](modernbert/RESULTS_COMPARISON.md) for latest results

### For Implementation Details
1. **ModernBERT**: [modernbert/EVALUATION_FINETUNED.md](modernbert/EVALUATION_FINETUNED.md)
2. **RoBERTa**: [roberta/IMPLEMENTATION_REPORT.md](roberta/IMPLEMENTATION_REPORT.md)
3. **AdaSVD**: [adasvd/README.md](adasvd/README.md)

### For Performance Analysis
1. [analysis/flashsvd/final-report.md](analysis/flashsvd/final-report.md)
2. [development/svdllm-v1-vs-v2-results.md](development/svdllm-v1-vs-v2-results.md)

---

## 📝 Document Status Legend

- ✅ **Active**: Current, actively maintained
- ✅ **Final**: Complete, final version
- ✅ **Reference**: Historical reference, accurate but superseded

---

## 🔄 Recent Updates

### 2026-02-11
- ✅ Added ModernBERT SVD-LLM v1 & v2 complete implementation
- ✅ Fine-tuned model evaluation (mrm8488/ModernBERT-base-ft-sst2)
- ✅ Validated v2 local update effectiveness (+0.56% improvement)
- ✅ Reorganized all encoder docs into eval_encoder/docs/

### Previous
- ✅ RoBERTa Whitening implementation and evaluation
- ✅ AdaSVD bug fixes and validation
- ✅ FlashSVD performance analysis
- ✅ Multiple benchmark result compilations

---

**For questions or updates, refer to the specific implementation directories.**
