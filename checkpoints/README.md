# Checkpoint Manifest

This directory tracks the local manifest for checkpoints used by the new LowRankArena scaffold.

## Source of Truth

- Hosted checkpoints live in the gated Hugging Face repository: `https://huggingface.co/Duke-CEI-SVD/LowRankArena/tree/main`
- At this stage, `index.csv` stores top-level folder entries observed from that repository.
- Individual exported variants, exact low-rank methods, and loader-specific metadata should be added incrementally.
- Optional sidecar manifests live under [`checkpoints/manifests/`](./manifests/README.md) when a checkpoint needs richer metadata than the flat CSV schema can express.

## `index.csv` Columns

- `name`: stable local identifier used by scripts
- `model_family`: coarse model family label
- `variant`: base, instruct, or another variant string
- `method`: compression or low-rank method label
- `source`: `huggingface` or `local`
- `repo_id`: remote model repository ID
- `revision`: git revision or HF branch name
- `subpath`: folder or file path inside the source repo
- `benchmarks`: pipe-separated benchmark tags
- `enabled`: whether the entry is active for default runs
- `notes`: free-form migration and bookkeeping notes

## Update Pattern

Use `scripts/add_checkpoint.py` to append or replace rows while keeping a consistent schema.

When a checkpoint is not a standard Transformers or vLLM load target, add a sidecar manifest first and keep the CSV row disabled by default. This preserves discoverability without polluting the default benchmark path with artifacts that still require export or a custom loader.

Locally generated artifacts from `compress/` should only be registered after they are exported into a loadable local checkpoint directory. Once that is true, register them with `source=local` and `subpath=<relative artifact dir>`, then use the normal benchmark scripts.
