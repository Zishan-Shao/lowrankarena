# `benchmark/accuracy/`

This directory contains accuracy-oriented benchmark suites.

All suites in this directory are specified in terms of exact `lm-eval-harness 0.4.11` task or group names. The Python runner is responsible for turning these configs into `lm-eval run ...` executions and normalizing the output format.

## Files

- [`mcq.yaml`](./mcq.yaml): multi-task multiple-choice suite for headline commonsense QA reporting.
- [`ppl.yaml`](./ppl.yaml): rolling loglikelihood suite for perplexity-style evaluation.
- [`mmlu.yaml`](./mmlu.yaml): dedicated MMLU suite using the official `mmlu` group.

## Conventions

- Use canonical `lm-eval` task IDs rather than project-local aliases.
- Record metric preference in the suite config so downstream scoring stays deterministic.
- Keep environment caveats in the suite that requires them, not in the runner.
