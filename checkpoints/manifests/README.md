# `checkpoints/manifests/`

This directory stores optional sidecar manifests for checkpoint entries that need richer metadata than the flat CSV registry can hold.

`index.csv` remains the runnable registry used by the CLI and benchmark runners. A sidecar manifest adds structured fields such as compression ratio, compatibility flags, base-model provenance, and loader notes without forcing the whole runtime onto a more complex schema.

Use this directory for:

- custom artifacts that are not standard Hugging Face `save_pretrained` packages,
- externally contributed checkpoints that need explicit provenance,
- future migrations from the flat CSV schema to a richer manifest-first model.

The default convention is one file per checkpoint ID:

- `<checkpoint_id>.json`

Use [`scripts/add_checkpoint.py`](../../scripts/add_checkpoint.py) or [`src/arena.py`](../../src/arena.py) to import a manifest and optionally persist its simplified runnable row into [`checkpoints/index.csv`](../index.csv).
