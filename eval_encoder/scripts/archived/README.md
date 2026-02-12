# Archived Test Scripts

This directory contains old test scripts that have been superseded by newer implementations.

## Contents

### Progress Checkers (Legacy)
- `check_progress.sh` - Old progress checker
- `check_retest_progress.sh` - Retest progress checker

**Superseded by**: `scripts/check_fp32_progress.sh`

### Test Runners (Legacy)
- `run_all_tests.sh` - Old comprehensive test runner
- `retest_all_e2e.sh` - End-to-end retest script
- `retest_e2e_priority1.sh` - Priority 1 retest script

**Superseded by**:
- `scripts/run_complete_fp32_benchmark.sh`
- `scripts/run_missing_tests_fp32.sh`

## Why Archived?

These scripts were used during the fp16 testing phase and early development. They have been replaced by more organized and comprehensive scripts in the main `scripts/` directory.

## Current Scripts

For current testing, use:
- `../run_complete_fp32_benchmark.sh` - Full benchmark suite
- `../run_missing_tests_fp32.sh` - Supplementary tests
- `../check_fp32_progress.sh` - Progress monitoring

---

**Archived**: 2026-02-12
