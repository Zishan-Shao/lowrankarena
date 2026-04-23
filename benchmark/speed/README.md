# `benchmark/speed/`

This directory contains speed benchmark specifications.

Speed suites are intentionally separate from accuracy suites because their runtime, metrics, and failure modes are different. These configs are executed by the suite-selected runner in [`src/speed_runner.py`](../../src/speed_runner.py): serving suites use `vLLM`, while `speed.yaml` measures the actual evaluation pipeline and defaults its nested `lm-eval` suites to the vLLM model backend.

## Files

- [`serve.yaml`](./serve.yaml): named leaderboard workload set for mainstream serving. The default cases cover interactive latency, decode-heavy generation, moderate context, and moderate throughput without drifting into stress-test territory.
- [`edge.yaml`](./edge.yaml): named edge-case workload set for long-context serving with `batch_size <= 4` and near-`16K` context budgets. This suite is meant for appendix or dedicated edge reporting rather than the default serving table.
- [`speed.yaml`](./speed.yaml): end-to-end evaluation-speed suite that runs the default base accuracy workloads and reports nested-suite wall-clock runtime rather than synthetic serving throughput.

## Conventions

- Treat prompt length, generation length, batch size, warmup, and repeat as first-class benchmark parameters.
- Prefer explicit named cases in paper-facing suites when you need reviewer-defensible coverage across qualitatively different serving regimes.
- Split mainstream serving and edge-case long-context workloads into separate suites so the serving leaderboard stays interpretable while edge behavior remains reproducible.
- Keep evaluation-speed separate from serving-speed: `speed.yaml` measures the benchmark pipeline, not `prefill/decode` microstructure. Its PPL child still uses the native contiguous runner; its `lm-eval` children default to `model_backend: vllm`.
- Keep the config focused on workload definition; device-specific tuning belongs in CLI overrides.
- Normalized speed outputs record `runtime.cuda_runtime` with selected GPU names, compute capability, total memory, and `CUDA_VISIBLE_DEVICES`; compare efficiency rows only within one GPU class.
- The `serve` defaults intentionally stay conservative enough to run on a partially occupied cluster. The `edge` suite assumes a less crowded GPU budget and therefore uses a higher default `gpu_memory_utilization`.
- Use CLI overrides on `scripts/run_speed.py` when you need to shrink or expand the sweep; serving suites accept batch/prompt/generation overrides, while evaluation-speed accepts `lm-eval` runtime overrides such as device, limit, or `--eval-model-backend hf`.
