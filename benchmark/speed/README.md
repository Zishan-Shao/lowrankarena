# `benchmark/speed/`

This directory contains speed benchmark specifications.

Speed suites are intentionally separate from accuracy suites because their runtime, metrics, and failure modes are different. These configs are executed by the vLLM-based runner in [`src/speed_runner.py`](../../src/speed_runner.py).

## Files

- [`speed.yaml`](./speed.yaml): default offline generation workload for latency and throughput reporting.

## Conventions

- Treat prompt length, generation length, batch size, warmup, and repeat as first-class benchmark parameters.
- Keep the config focused on workload definition; device-specific tuning belongs in CLI overrides.
