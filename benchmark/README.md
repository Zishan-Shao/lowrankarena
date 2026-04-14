# `benchmark/`

This directory defines the public benchmark surface of LowRankArena.

It contains declarative suite specifications only. Execution logic lives in [`scripts/`](../scripts/README.md) and reusable runtime code lives in [`src/`](../src/README.md).

## Structure

- [`main.yaml`](./main.yaml): aggregate entrypoint that expands into the default paper-facing workload, including `WikiText-2` and `C4` perplexity, headline MCQ accuracy, `MMLU-Pro`, `GSM8K`, memory, and speed.
- [`accuracy/`](./accuracy/README.md): accuracy suites that select their own backend per suite. Most classification suites still use `lm-eval-harness 0.4.11`, while `ppl` now uses a repo-owned contiguous perplexity runner so preprocessing stays fixed across checkpoints.
- [`speed/`](./speed/README.md): offline inference speed suites backed by vLLM.

Memory is currently handled by the dedicated CLI path in [`scripts/run_memory.py`](../scripts/run_memory.py) rather than a YAML suite tree under `benchmark/`.

## Design Intent

- Keep benchmark definitions human-readable and reviewable.
- Make suite membership and task selection explicit in version-controlled YAML.
- Avoid embedding runner logic in configuration files.
