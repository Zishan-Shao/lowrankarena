# `scripts/`

This directory contains thin command-line entrypoints for the benchmark-only
artifact. Each script parses arguments, resolves the requested suite or
checkpoint, and delegates execution to [`src/`](../src/README.md).

## Entry Points

- [`run_eval.py`](./run_eval.py): run one accuracy/perplexity suite for one checkpoint.
- [`run_main.py`](./run_main.py): execute aggregate evaluation lanes such as `base` and `instruct`.
- [`run_memory.py`](./run_memory.py): run the active-memory suite for one checkpoint.
- [`run_speed.py`](./run_speed.py): run serving, online serving, or evaluation-speed suites.
- [`run_all.py`](./run_all.py): compatibility alias for `run_main.py`.
- [`measure_peak_memory.py`](./measure_peak_memory.py): compatibility alias for `run_memory.py`.
- [`make_table.py`](./make_table.py): build lightweight tables from normalized result JSON files.
- [`add_checkpoint.py`](./add_checkpoint.py): update [`checkpoints/index.csv`](../checkpoints/index.csv) or import a sidecar manifest.

Benchmark-critical protocol settings belong in suite YAML files under
[`benchmark/`](../benchmark/README.md), not in ad hoc shell snippets.

By default, benchmark suites use `dtype: auto`, which follows the checkpoint's
native fp16 or bf16 precision. CLI overrides accept aliases such as `fp16`,
`float16`, `bf16`, and `bfloat16`.

The `lm-eval` accuracy suites declare `model_backend: vllm` in YAML for faster
full runs. Pass `--model-backend hf` on `run_eval.py` or
`--eval-model-backend hf` on `run_main.py` for a Transformers/HF comparison.
The contiguous PPL suite uses its native runner regardless of this setting.
