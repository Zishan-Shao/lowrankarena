# Benchmark Scripts

Essential scripts for encoder SVD compression benchmark analysis and data management.

---

## 📁 Active Scripts

### Analysis & Reporting
- **generate_readme_table_fp32.py** - Generate markdown performance table from CSV results
- **check_fp32_progress.sh** - Monitor progress of running benchmark tests

### Data Management
- **organize_csvs.py** - Organize and consolidate CSV result files

---

## 🗄️ Archived Scripts (`archived/`)

All specialized test scripts have been archived after completion of the comprehensive FP32 benchmark suite (45 configurations, 2026-02-12):

- **core/** - Core benchmark tests (small ranks, long sequences, memory analysis)
- **adasvd/** - AdaSVD-specific test scripts (5 budgets × 2 backends)
- **sweeps/** - Parameter sweep scripts (long sequence tests)
- **Legacy** - Old test runners and progress checkers

These archived scripts are preserved for reference and can be adapted for future testing needs.

---

## 🚀 Quick Usage

### Generate README Table

After running benchmarks, update the README table:

```bash
cd eval_encoder/scripts
python generate_readme_table_fp32.py
```

Reads from: `eval_encoder/eval_results/encoder_runs.csv`
Outputs: Markdown table for README.md

### Check Test Progress

Monitor running benchmarks:

```bash
cd eval_encoder/scripts
./check_fp32_progress.sh
```

Shows current test status and completion estimates.

### Organize Results

Consolidate and archive result files:

```bash
cd eval_encoder/scripts
python organize_csvs.py
```

Organizes files in `eval_encoder/eval_results/`.

---

## 📊 Current Benchmark Results

The comprehensive FP32 benchmark (45 configurations) includes:

- **Dense baseline**: 1 config
- **SVD**: 10 configs (5 ranks × 2 backends)
- **FWSVD**: 10 configs (5 ranks × 2 backends)
- **DRONE**: 10 configs (5 ranks × 2 backends)
- **AdaSVD**: 14 configs (7 budgets × 2 backends)

All results are stored in `eval_encoder/eval_results/encoder_runs.csv`.

---

## 🔗 Related Documentation

- [Main README](../README.md) - eval_encoder overview and results
- [Getting Started](../docs/guides/getting-started.md) - Quick start guide
- [Benchmark Guide](../docs/guides/benchmark-guide.md) - Complete testing guide

---

**Last Updated**: 2026-02-12
