# Benchmark Scripts

**Organized test scripts for encoder SVD compression benchmarks.**

---

## 📁 Directory Structure

```
scripts/
├── README.md              # This file
│
├── core/                  # Core benchmark tests (general purpose)
│   ├── check_progress.sh                # Monitor test progress
│   ├── test_flashsvd_memory.sh          # FlashSVD memory efficiency
│   ├── test_longseq_lowrank.sh          # Long sequence tests
│   ├── test_small_ranks.sh              # Small rank analysis (R=16-64)
│   ├── test_small_ranks_complete.sh     # Complete small rank sweep
│   ├── test_extreme_memory.sh           # Extreme memory tests
│   └── test_extreme_fwsvd_ada.sh        # FWSVD vs AdaSVD comparison
│
├── adasvd/                # AdaSVD-specific tests
│   ├── test_adasvd_5budgets.sh          # ⭐ Main: 5 budgets × 2 backends
│   ├── test_adasvd.sh                   # Standard AdaSVD test
│   ├── test_adasvd_fixed.sh             # AdaSVD with bug fixes
│   ├── test_adasvd_flashsvd_all.sh      # All budgets, FlashSVD only
│   ├── test_adasvd_naive_all.sh         # All budgets, naive only
│   ├── retest_adasvd_all.sh             # Rerun all AdaSVD tests
│   └── check_adasvd_progress.sh         # AdaSVD-specific progress check
│
├── sweeps/                # Parameter sweep scripts
│   └── test_flashsvd_longseq.sh         # FlashSVD long sequence sweep
│
└── utils/                 # Utility scripts
    └── organize_csvs.py                 # Organize and consolidate results
```

---

## 🚀 Quick Start

### 1. AdaSVD Comprehensive Test (Recommended)

Test 5 budgets (0.3, 0.4, 0.5, 0.6, 0.7) × 2 backends (naive + flashsvd) = 10 runs

```bash
cd eval_encoder/scripts/adasvd
./test_adasvd_5budgets.sh
```

**Duration**: ~40-50 minutes
**Output**: `eval_results/encoder_runs_sst2_adasvd_refactored_5budgets.csv`

**What it tests:**
- Budget control accuracy (10%, 30%, 50%, etc.)
- Naive vs FlashSVD backend comparison
- Memory efficiency across compression levels
- Throughput and accuracy trade-offs

---

### 2. FlashSVD Memory Efficiency

Test FlashSVD memory savings with various methods and ranks.

```bash
cd eval_encoder/scripts/core
./test_flashsvd_memory.sh
```

**Duration**: ~20-25 minutes
**Output**: `eval_results/encoder_runs_sst2_flashsvd_memory.csv`

**What it tests:**
- SVD rank=32: naive vs flashsvd
- FWSVD rank=32: naive vs flashsvd
- DRONE rank=32: naive vs flashsvd
- Memory consumption and throughput

---

### 3. Long Sequence Tests

Test memory scaling with extended sequence lengths (seq=128 to seq=1024).

```bash
cd eval_encoder/scripts/core
./test_longseq_lowrank.sh
```

**Duration**: ~30-40 minutes
**Output**: `eval_results/encoder_runs_longseq_lowrank.csv`

**What it tests:**
- Memory growth with sequence length
- FlashSVD memory advantages at scale
- Throughput impact of longer sequences

---

### 4. Small Ranks Analysis

Analyze behavior at aggressive compression (R=16, 32, 48, 64).

```bash
cd eval_encoder/scripts/core
./test_small_ranks_complete.sh
```

**Duration**: ~15-20 minutes
**Output**: `eval_results/encoder_runs_sst2_small_ranks.csv`

**What it tests:**
- Accuracy degradation at low ranks
- Memory-accuracy trade-off curve
- Minimum viable rank for production

---

## 📊 Script Categories

### Core Tests (General Purpose)

| Script | Purpose | Duration | Output CSV |
|--------|---------|----------|------------|
| `check_progress.sh` | Monitor running tests | Instant | N/A |
| `test_flashsvd_memory.sh` | FlashSVD memory efficiency | 20-25 min | `flashsvd_memory.csv` |
| `test_longseq_lowrank.sh` | Long sequence scaling | 30-40 min | `longseq_lowrank.csv` |
| `test_small_ranks.sh` | Small rank sweep (quick) | 8-10 min | `small_ranks.csv` |
| `test_small_ranks_complete.sh` | Complete rank analysis | 15-20 min | `small_ranks.csv` |
| `test_extreme_memory.sh` | Extreme configurations | 25-30 min | `extreme_memory.csv` |
| `test_extreme_fwsvd_ada.sh` | FWSVD vs AdaSVD | 15-20 min | `extreme_fwsvd_ada.csv` |

### AdaSVD Tests

| Script | Purpose | Duration | Output CSV |
|--------|---------|----------|------------|
| `test_adasvd_5budgets.sh` | ⭐ Main test: 5×2=10 runs | 40-50 min | `adasvd_refactored_5budgets.csv` |
| `test_adasvd.sh` | Single AdaSVD test | 4-5 min | `adasvd_test.csv` |
| `test_adasvd_fixed.sh` | Fixed budget control | 4-5 min | `adasvd_fixed.csv` |
| `test_adasvd_flashsvd_all.sh` | FlashSVD only, all budgets | 20-25 min | `adasvd_flashsvd_all.csv` |
| `test_adasvd_naive_all.sh` | Naive only, all budgets | 20-25 min | `adasvd_naive_all.csv` |
| `retest_adasvd_all.sh` | Rerun all AdaSVD tests | 40-50 min | Multiple files |
| `check_adasvd_progress.sh` | Check AdaSVD test status | Instant | N/A |

### Parameter Sweeps

| Script | Purpose | Duration | Output CSV |
|--------|---------|----------|------------|
| `test_flashsvd_longseq.sh` | FlashSVD long sequence sweep | 30-40 min | `flashsvd_longseq.csv` |

### Utilities

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `organize_csvs.py` | Consolidate results | `eval_results/*.csv` | `consolidated/` |

---

## 🔧 Usage Notes

### Running Scripts

All scripts should be run from the **eval_encoder/scripts/** directory or its subdirectories:

```bash
# Correct
cd eval_encoder/scripts/adasvd
./test_adasvd_5budgets.sh

# Also correct
cd eval_encoder/scripts/core
./test_flashsvd_memory.sh
```

### Monitoring Progress

Use `check_progress.sh` to monitor running tests:

```bash
cd eval_encoder/scripts/core
./check_progress.sh
```

This will show:
- Current running process
- Estimated completion time
- Result files being generated

### Organizing Results

After running tests, consolidate results:

```bash
cd eval_encoder/scripts/utils
python organize_csvs.py
```

This creates organized result files in `eval_results/consolidated/`.

---

## 📂 Output Files

All test scripts output CSV files to `eval_encoder/eval_results/`.

### File Naming Convention

- `encoder_runs_<task>_<method>_<config>.csv` - Individual test results
- `consolidated/<method>_benchmarks.csv` - Consolidated results by method

### CSV Columns

Standard columns in result files:
- `method`: Compression method (dense, svd, fwsvd, drone, adasvd)
- `backend`: Execution backend (naive, flashsvd)
- `rank` or `budget`: Compression parameter
- `metric_value`: Task accuracy
- `peak_mem_mb`: Peak memory usage
- `latency_ms`: Inference latency
- `samples_per_sec`: Throughput

---

## 🎯 Common Workflows

### 1. Full Benchmark Suite (1-2 hours)

```bash
cd eval_encoder/scripts

# AdaSVD comprehensive test
adasvd/test_adasvd_5budgets.sh

# FlashSVD memory efficiency
core/test_flashsvd_memory.sh

# Long sequence scaling
core/test_longseq_lowrank.sh

# Small ranks analysis
core/test_small_ranks_complete.sh

# Consolidate results
cd utils && python organize_csvs.py
```

### 2. Quick Sanity Check (5-10 minutes)

```bash
cd eval_encoder/scripts

# Single AdaSVD test
adasvd/test_adasvd.sh

# Quick small ranks test
core/test_small_ranks.sh
```

### 3. AdaSVD Deep Dive (40-50 minutes)

```bash
cd eval_encoder/scripts/adasvd

# Main test
./test_adasvd_5budgets.sh

# Check progress during run
./check_adasvd_progress.sh

# Analyze results
cd ../../
python analyze_ranks.py
```

---

## 🐛 Troubleshooting

### Script fails with "command not found"

Make sure you're in the correct directory and the script is executable:

```bash
cd eval_encoder/scripts/<category>
chmod +x <script>.sh
./<script>.sh
```

### Out of memory errors

Reduce batch size or sequence length in the script, or use FlashSVD backend:

```bash
# Edit script to change:
--batch_size 32   # → --batch_size 16
--seq_len 128     # → --seq_len 64
--backend flashsvd  # Use memory-efficient backend
```

### Results not appearing in CSV

Check for errors in the script output. The CSV is only written if the test completes successfully.

---

## 📝 Adding New Scripts

When adding new test scripts:

1. **Choose the right category**:
   - Core tests → `core/`
   - AdaSVD-specific → `adasvd/`
   - Parameter sweeps → `sweeps/`
   - Utilities → `utils/`

2. **Follow naming convention**:
   - `test_<feature>_<config>.sh` for test scripts
   - `check_<what>.sh` for monitoring scripts
   - `<action>_<target>.py` for utilities

3. **Update this README** with:
   - Script description
   - Expected duration
   - Output file name

---

## 🔗 Related Documentation

- **[Getting Started](../docs/guides/getting-started.md)** - Overview and quick start
- **[Benchmark Guide](../docs/guides/benchmark-guide.md)** - Complete testing guide
- **[Quick Reference](../docs/guides/quick-reference.md)** - FlashSVD cheat sheet
- **[Main README](../README.md)** - eval_encoder overview

---

**Last Updated**: 2026-02-10
