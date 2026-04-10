# `benchmark/`

This directory defines the public benchmark surface of LowRankArena.

It contains declarative suite specifications only. Execution logic lives in [`scripts/`](../scripts/README.md) and reusable runtime code lives in [`src/`](../src/README.md).

## Structure

- [`main.yaml`](./main.yaml): aggregate entrypoint that expands into the default benchmark workload.
- [`accuracy/`](./accuracy/README.md): accuracy suites backed by `lm-eval-harness 0.4.11`.
- [`speed/`](./speed/README.md): offline inference speed suites backed by vLLM.

## Design Intent

- Keep benchmark definitions human-readable and reviewable.
- Make suite membership and task selection explicit in version-controlled YAML.
- Avoid embedding runner logic in configuration files.
