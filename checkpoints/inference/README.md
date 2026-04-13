# `checkpoints/inference/`

This directory documents the layout for materialized Transformers-compatible inference wrappers.

These wrappers are derived artifacts, not source checkpoints. The runtime now defaults to an external cache under `~/.cache/lowrankarena/inference`, so this folder should stay empty unless a wrapper is intentionally archived for inspection or release packaging.

## Intended Content

- Wrapper directories produced by [`src/inference_adapter.py`](../../src/inference_adapter.py) or related materialization utilities.
- Metadata files that explain why a particular checkpoint needs an inference wrapper.

## Notes

- Keep benchmark outputs in [`results/`](../../results/README.md), not here.
- Keep runnable checkpoint metadata in [`checkpoints/index.csv`](../index.csv), not in local cache folders.
