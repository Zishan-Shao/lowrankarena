# `scripts/`

This directory contains the user-facing command-line entrypoints for LowRankArena.

Each script is intentionally thin. It should parse arguments, resolve the requested suite or checkpoint, and delegate real work to [`src/`](../src/README.md).

## Entry Points

- [`run_eval.py`](./run_eval.py): run a single accuracy suite against one checkpoint through `lm-eval-harness`.
- [`run_memory.py`](./run_memory.py): run a single Transformers-based memory measurement for one checkpoint.
- [`run_speed.py`](./run_speed.py): run a single vLLM speed suite against one checkpoint.
- [`measure_peak_memory.py`](./measure_peak_memory.py): compatibility alias for `run_memory.py`.
- [`run_main.py`](./run_main.py): execute aggregate benchmark suites such as [`benchmark/main.yaml`](../benchmark/main.yaml).
- [`run_all.py`](./run_all.py): compatibility alias for `run_main.py`.
- [`run_compress.py`](./run_compress.py): plan or dispatch optional artifact-generation flows from [`compress/`](../compress/README.md).
- [`make_table.py`](./make_table.py): build lightweight table artifacts from normalized result JSON files.
- [`add_checkpoint.py`](./add_checkpoint.py): update [`checkpoints/index.csv`](../checkpoints/index.csv) or import a sidecar manifest from [`checkpoints/manifests/`](../checkpoints/manifests/README.md).

## Design Intent

- Keep CLI behavior explicit and inspectable.
- Avoid duplicating business logic across multiple scripts.
- Preserve a stable user interface even as runners evolve.
