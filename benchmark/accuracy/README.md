# `benchmark/accuracy/`

This directory contains accuracy-oriented benchmark suites.

All suites in this directory are specified in terms of exact `lm-eval-harness 0.4.11` task or group names. The Python runner turns these configs into `lm-eval run ...` executions through [`scripts/run_eval.py`](../../scripts/run_eval.py) and [`src/lm_eval_runner.py`](../../src/lm_eval_runner.py), then writes normalized outputs to [`results/eval/`](../../results/eval/README.md).

## Files

- [`mcq.yaml`](./mcq.yaml): multi-task multiple-choice suite for headline commonsense QA reporting.
- [`ppl.yaml`](./ppl.yaml): rolling loglikelihood suite for perplexity-style evaluation.
- [`mmlu.yaml`](./mmlu.yaml): dedicated MMLU suite using the official `mmlu` group.

## Conventions

- Use canonical `lm-eval` task IDs rather than project-local aliases.
- Record metric preference in the suite config so downstream scoring stays deterministic.
- Keep environment caveats in the suite that requires them, not in the runner.
- Keep suite files focused on task selection and metric intent; runtime flags belong in the CLI.
