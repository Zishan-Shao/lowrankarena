# Documentation Index

**Complete documentation for the Encoder SVD Benchmark pipeline.**

---

## 📖 User Guides

Essential documentation for getting started and running benchmarks.

### [Getting Started](guides/getting-started.md)
- Supported architectures (BERT, RoBERTa, ModernBERT)
- Execution modes (dense, naive, FlashSVD)
- Quick start examples
- Command-line arguments reference

### [Benchmark Guide](guides/benchmark-guide.md)
- Directory structure overview
- Complete testing workflows
- Result consolidation and analysis
- Best practices and troubleshooting

### [Quick Reference](guides/quick-reference.md)
- FlashSVD configuration cheat sheet
- Decision flowchart for backend selection
- Performance expectations by scenario
- Common pitfalls and solutions

---

## 📊 Performance Analysis

In-depth analysis reports on compression methods and optimization strategies.

### FlashSVD Reports

#### [Final Report](analysis/flashsvd/final-report.md)
Comprehensive evaluation of FlashSVD across all methods and configurations.

**Key Findings:**
- 65-72% memory reduction vs naive backend
- 32-48% throughput improvement for large batches
- Best config: DRONE rank=32 @ seq=512, batch=64

#### [Performance Analysis](analysis/flashsvd/performance-analysis.md)
Detailed benchmarks comparing naive vs FlashSVD backends.

**Covers:**
- Memory efficiency across sequence lengths
- Throughput scaling with batch size
- Rank-dependent behavior
- Trade-off analysis

#### [Long Sequence Results](analysis/flashsvd/longseq-results.md)
Memory efficiency testing at extended sequence lengths (seq=128 to seq=1024).

**Highlights:**
- Memory savings increase with sequence length
- FlashSVD enables 2× longer sequences within same memory budget
- Critical for document-level and long-context tasks

#### [DRONE Highlight](analysis/flashsvd/drone-highlight.md)
DRONE method performance with FlashSVD optimization.

**Key Metrics:**
- 72% memory reduction
- 9% latency improvement
- 71% accuracy maintained

### [Small Ranks Analysis](analysis/small-ranks-analysis.md)
Behavior analysis at low compression ranks (R=16-48).

**Findings:**
- Rank 16: Significant accuracy drop, limited practical use
- Rank 32: Sweet spot for aggressive compression
- Rank 48: Balanced compression-accuracy trade-off

---

## 🔧 Development & Debugging

Technical documentation on bug fixes, implementation details, and debugging notes.

### [AdaSVD Bug Analysis](development/adasvd-bug-analysis.md)
Investigation of budget control bug in original AdaSVD implementation.

**Problem Identified:**
- All budgets converged to 66.5% regardless of target
- Root causes: incorrect base calculation, one-sided penalty, loss imbalance

**Impact:**
- Budget 0.1 → actual 66.5% (should be 10%)
- Budget 0.5 → actual 66.5% (should be 50%)

### [AdaSVD Fixes Applied](development/adasvd-fixes-applied.md)
Implementation details of the budget control fix.

**Changes Made:**
1. Fixed budget base calculation (use M×N instead of (M+N)×R)
2. Implemented two-sided squared error penalty
3. Rebalanced loss weights (100.0×budget + 0.01×align)

**Verification:**
- Budget 0.1 → 10.2% ✅
- Budget 0.3 → 29.6% ✅
- Budget 0.5 → 49.0% ✅

**FlashSVD Compatibility:**
- Median rank strategy for budgets < 0.3
- Adaptive ranks for budgets ≥ 0.3
- 70% speedup at budget=0.1 with FlashSVD

---

## 📂 Document Organization

```
docs/
├── README.md                           # This file
│
├── guides/                             # User documentation
│   ├── getting-started.md              # Quick start guide
│   ├── benchmark-guide.md              # Complete testing guide
│   └── quick-reference.md              # Cheat sheet
│
├── analysis/                           # Performance reports
│   ├── flashsvd/
│   │   ├── final-report.md             # Comprehensive evaluation
│   │   ├── performance-analysis.md     # Detailed benchmarks
│   │   ├── longseq-results.md          # Memory at scale
│   │   └── drone-highlight.md          # DRONE performance
│   └── small-ranks-analysis.md         # Low-rank behavior
│
└── development/                        # Dev notes
    ├── adasvd-bug-analysis.md          # Bug investigation
    └── adasvd-fixes-applied.md         # Fix implementation
```

---

## 🔄 Document Status

| Document | Status | Last Updated |
|----------|:------:|:------------:|
| Getting Started | ✅ Stable | 2026-02-08 |
| Benchmark Guide | ✅ Stable | 2026-02-09 |
| Quick Reference | ✅ Stable | 2026-02-09 |
| FlashSVD Final Report | ✅ Complete | 2026-02-09 |
| FlashSVD Performance | ✅ Complete | 2026-02-09 |
| FlashSVD Long Seq | ✅ Complete | 2026-02-09 |
| DRONE Highlight | ✅ Complete | 2026-02-09 |
| Small Ranks | ✅ Complete | 2026-02-09 |
| AdaSVD Bug Analysis | ✅ Complete | 2026-02-09 |
| AdaSVD Fixes | ✅ Complete | 2026-02-09 |

---

## 🎯 Quick Navigation

**New Users**: Start with [Getting Started](guides/getting-started.md)
**Running Benchmarks**: See [Benchmark Guide](guides/benchmark-guide.md)
**Need Quick Answer**: Check [Quick Reference](guides/quick-reference.md)
**Optimization Help**: Read [FlashSVD Final Report](analysis/flashsvd/final-report.md)
**Debugging AdaSVD**: See [Development Docs](development/)

---

**Parent**: [eval_encoder/README.md](../README.md)
