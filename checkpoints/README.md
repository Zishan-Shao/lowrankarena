# `checkpoints/`

This directory tracks the runnable checkpoint registry and related checkpoint
metadata used by the benchmark runners.

## Source Of Truth

- [`index.csv`](./index.csv) is the runnable registry used by the CLI and benchmark runners.
- The included low-rank rows are anonymized placeholders. Replace their `repo_id`
  and `subpath`, or switch them to `source=local`, before running them.
- Optional sidecar manifests live under [`manifests/`](./manifests/README.md)
  when a checkpoint needs richer metadata than the flat CSV schema can express.
- Materialized inference wrappers can be archived under [`inference/`](./inference/README.md).
- Materialized vLLM-compatible wrapper checkpoints can be archived under [`vllm/`](./vllm/README.md).

Runtime caches live outside the git repository under `~/.cache/lowrankarena/`
unless overridden with environment variables such as `LRA_INFERENCE_CACHE_ROOT`
and `LRA_VLLM_CACHE_ROOT`.

## `index.csv` Columns

- `name`: stable local identifier used by scripts
- `model_family`: coarse model family label
- `variant`: base, instruct, or another variant string
- `method`: compression or low-rank method label
- `source`: `huggingface` or `local`
- `repo_id`: remote model repository ID, blank for local rows
- `revision`: git revision or HF branch name
- `subpath`: folder or file path inside the source repo, or local path for `source=local`
- `benchmarks`: pipe-separated benchmark tags
- `enabled`: whether the entry is active for default runs
- `notes`: free-form metadata notes

## Update Pattern

Use [`scripts/add_checkpoint.py`](../scripts/add_checkpoint.py) to append or
replace rows while keeping a consistent schema.

For anonymous review, avoid committing rows that reveal private organization
names, usernames, absolute home paths, or non-anonymous hosted checkpoint repos.
