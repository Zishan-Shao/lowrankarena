# `scripts/`

This directory contains the user-facing command-line entrypoints for LowRankArena.

Each script is intentionally thin. It should parse arguments, resolve the requested suite or checkpoint, and delegate real work to [`src/`](../src/README.md). Benchmark-critical lm-eval protocol settings belong in suite YAML rather than in handwritten shell snippets.

## Entry Points

- [`run_eval.py`](./run_eval.py): run a single accuracy suite against one checkpoint through the suite-selected eval backend.
- [`run_memory.py`](./run_memory.py): run a dedicated Transformers-based memory suite for one checkpoint. It defaults to [`benchmark/memory/active.yaml`](../benchmark/memory/active.yaml).
- [`run_speed.py`](./run_speed.py): run a dedicated speed suite against one checkpoint. Serving suites use `vLLM`; `speed/speed` measures end-to-end evaluation runtime.
- [`measure_peak_memory.py`](./measure_peak_memory.py): compatibility alias for `run_memory.py`.
- [`run_main.py`](./run_main.py): execute evaluation lanes such as [`benchmark/base.yaml`](../benchmark/base.yaml) and [`benchmark/instruct.yaml`](../benchmark/instruct.yaml). It defaults to the paper-facing Base/Universal lane, while memory and speed stay on their dedicated scripts.
- [`run_all.py`](./run_all.py): compatibility alias for `run_main.py`.
- [`run_compress.py`](./run_compress.py): plan or dispatch optional artifact-generation flows from [`compress/`](../compress/README.md).
- [`make_table.py`](./make_table.py): build lightweight table artifacts from normalized result JSON files.
- [`add_checkpoint.py`](./add_checkpoint.py): update [`checkpoints/index.csv`](../checkpoints/index.csv) or import a sidecar manifest from [`checkpoints/manifests/`](../checkpoints/manifests/README.md).

By default, benchmark suites use `dtype: auto`, which follows the checkpoint's native fp16 or bf16 precision. CLI overrides accept common aliases such as `fp16`, `float`, `bf16`, and `bfloat`.

Paper-facing `lm-eval` accuracy suites declare `model_backend: vllm` in YAML for faster full runs. Pass `--model-backend hf` on `run_eval.py` or `--eval-model-backend hf` on `run_main.py` for a Transformers/HF comparison. The contiguous PPL suite uses its native runner regardless of this setting. `speed/speed` also uses vLLM for its nested `lm-eval` suites unless overridden with `--eval-model-backend hf`.

## Design Intent

- Keep CLI behavior explicit and inspectable.
- Avoid duplicating business logic across multiple scripts.
- Preserve a stable user interface even as runners evolve.
- Keep filesystem outputs predictable: checkpoint artifacts belong in `checkpoints/`, and benchmark outputs belong in `results/`.
