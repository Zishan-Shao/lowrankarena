# `tests/`

This directory contains lightweight regression tests for repository structure and runner-adjacent behavior.

The current test suite is intentionally pragmatic. It focuses on schema, layout, and small pieces of logic that should remain stable while the benchmark evolves.

## Files

- [`test_manifest.py`](./test_manifest.py): verify expected repository layout and seed checkpoint rows.
- [`test_load.py`](./test_load.py): validate checkpoint loading helpers.
- [`test_eval.py`](./test_eval.py): validate small eval-runner helper behavior.
- [`test_compress.py`](./test_compress.py): validate compression-planning scaffolding.
- [`test_benchmark_configs.py`](./test_benchmark_configs.py): verify benchmark configs stay aligned with exact `lm-eval` task names.

## Scope

- Fast structural checks.
- No requirement to reproduce full benchmark workloads inside unit tests.
