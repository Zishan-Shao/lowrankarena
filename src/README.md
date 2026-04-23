# `src/`

This directory contains the reusable runtime code for LowRankArena.

The guiding rule is simple: benchmark policy lives in [`benchmark/`](../benchmark/README.md), CLI entrypoints live in [`scripts/`](../scripts/README.md), and reusable execution logic lives here.

## Core Modules

- [`arena.py`](./arena.py): minimal public facade for listing, registering, evaluating, and reporting checkpoints.
- [`modeling/`](./modeling): shared low-rank runtime grouped by supported base-model family.
  The current family groups are [`llama/`](./modeling/llama/README.md), [`mistral/`](./modeling/mistral/README.md), and [`qwen/`](./modeling/qwen/README.md).
- [`vllm/`](./vllm/README.md): vLLM-specific adapter code and prototype utilities for wrapper-based loading.
- [`load.py`](./load.py): resolve checkpoint records into local or Hugging Face load targets.
- [`loader.py`](./loader.py): compatibility re-export for code that expects a `loader` module name.
- [`registry.py`](./registry.py): read, filter, and update checkpoint manifest records.
- [`benchmarking.py`](./benchmarking.py): resolve suite paths and select checkpoints for a suite.
- [`lm_eval_runner.py`](./lm_eval_runner.py): thin wrapper around `lm-eval run ...`, plus result normalization.
- [`ppl_runner.py`](./ppl_runner.py): project-owned contiguous-block perplexity runner for unified local PPL evaluation.
- [`memory_runner.py`](./memory_runner.py): Transformers-based active-memory measurement runner.
- [`speed_runner.py`](./speed_runner.py): suite-selected speed runner for both vLLM serving workloads and evaluation-speed workloads.
- [`hardware.py`](./hardware.py): CUDA device metadata helpers used to record GPU model/class in efficiency outputs.
- [`scoring.py`](./scoring.py): task-level metric selection and summary reduction helpers.
- [`report.py`](./report.py): lightweight result discovery and table generation.
- [`eval.py`](./eval.py): compatibility wrapper that forwards eval requests to the real runner.
- [`memory.py`](./memory.py): compatibility wrapper that forwards memory requests to the real runner.
- [`result_schema.py`](./result_schema.py): shared top-level JSON schema builder for normalized benchmark outputs.
- [`speed.py`](./speed.py): compatibility wrapper that forwards speed requests to the real runner.
- [`utils.py`](./utils.py): shared filesystem, JSON, YAML, and timestamp helpers.

## Design Intent

- Keep the public API thin: `Arena` is the only intended entrypoint for programmatic use.
- Support methods through a shared family-level runtime, not through per-checkpoint custom forward definitions.
- Keep external dependencies at the boundary.
- Normalize raw backend output into a project-owned JSON shape.
- Make it easy to swap runners without rewriting every CLI entrypoint.
- Keep checkpoint-compatibility logic in adapters, not in the benchmark runners themselves.
