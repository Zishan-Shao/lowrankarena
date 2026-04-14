# `tests/`

This directory contains lightweight regression tests for repository structure and runner-adjacent behavior.

The current test suite is intentionally pragmatic. It focuses on schema, layout, and small pieces of logic that should remain stable while the benchmark evolves.

## Files

- [`test_manifest.py`](./test_manifest.py): verify expected repository layout and seed checkpoint rows.
- [`test_modeling.py`](./test_modeling.py): validate family-level modeling wrappers install shared low-rank layers.
- [`test_ppl_runner.py`](./test_ppl_runner.py): validate contiguous-block perplexity helper behavior.
- [`test_load.py`](./test_load.py): validate checkpoint loading helpers.
- [`test_eval.py`](./test_eval.py): validate small eval-runner helper behavior.
- [`test_memory.py`](./test_memory.py): validate memory-runner helper behavior.
- [`test_result_schema.py`](./test_result_schema.py): validate the shared normalized result shape.
- [`test_vllm_adapter.py`](./test_vllm_adapter.py): validate the stable adapter contract for vLLM-prepared checkpoints.
- [`test_compress.py`](./test_compress.py): validate compression-planning scaffolding.
- [`test_benchmark_configs.py`](./test_benchmark_configs.py): verify benchmark configs stay aligned with exact `lm-eval` task names.

## Scope

- Fast structural checks.
- No requirement to reproduce full benchmark workloads inside unit tests.
- Prefer pure-Python helper coverage over GPU-heavy end-to-end tests.
