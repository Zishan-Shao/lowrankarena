# `results/eval/`

This directory stores normalized accuracy results produced by the `lm-eval` runner.

The files here are project-owned summaries, not raw `lm-eval` dumps. Raw backend outputs may be stored in a separate runtime-created subdirectory when requested by the runner.

## Expected Content

- One JSON file per `suite x checkpoint`.
- Shared top-level schema plus eval-specific normalized task metrics in `metrics` and task-level detail in `details`.
