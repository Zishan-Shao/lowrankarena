# `benchmark/speed/`

This directory contains speed benchmark specifications.

Speed suites are intentionally separate from accuracy suites because their runtime, metrics, and failure modes are different. These configs are executed by the suite-selected runner in [`src/speed_runner.py`](../../src/speed_runner.py): serving suites use `vLLM`, while `speed.yaml` measures the actual evaluation pipeline.

## Files

- [`serve.yaml`](./serve.yaml): named leaderboard workload set for mainstream serving. The default cases cover interactive latency, decode-heavy generation, moderate context, and moderate throughput without drifting into stress-test territory.
- [`edge.yaml`](./edge.yaml): named edge-case workload set for long-context serving with `batch_size <= 4` and near-`16K` context budgets. This suite is meant for appendix or dedicated edge reporting rather than the default main table.
- [`speed.yaml`](./speed.yaml): end-to-end evaluation-speed suite that runs the project-owned accuracy workloads and reports nested-suite wall-clock runtime rather than synthetic serving throughput.

## Conventions

- Treat prompt length, generation length, batch size, warmup, and repeat as first-class benchmark parameters.
- Prefer explicit named cases in paper-facing suites when you need reviewer-defensible coverage across qualitatively different serving regimes.
- Split mainstream serving and edge-case long-context workloads into separate suites so the main leaderboard stays interpretable while edge behavior remains reproducible.
- Keep evaluation-speed separate from serving-speed: `speed.yaml` measures the benchmark pipeline, not `prefill/decode` microstructure.
- Keep the config focused on workload definition; device-specific tuning belongs in CLI overrides.
- The `serve` defaults intentionally stay conservative enough to run on a partially occupied cluster. The `edge` suite assumes a less crowded GPU budget and therefore uses a higher default `gpu_memory_utilization`.
- Use CLI overrides on `scripts/run_speed.py` or `scripts/run_main.py` when you need to shrink or expand the sweep; serving suites accept batch/prompt/generation overrides, while evaluation-speed accepts `lm-eval` runtime overrides such as device or limit.
