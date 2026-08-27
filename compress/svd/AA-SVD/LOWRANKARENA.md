# AA-SVD in LowRankArena

This directory is a source-only snapshot of
[AA-SVD](https://github.com/atulkumarin/AA-SVD) with the compatibility changes
used by LowRankArena. It is included for artifact generation and source audit;
the normal benchmark path does not require recompressing a model.

## Provenance

- Upstream commit: [`1fa1b686cd9b13a77607a676564e37d438a176c8`](https://github.com/atulkumarin/AA-SVD/commit/1fa1b686cd9b13a77607a676564e37d438a176c8)
- Upstream license: MIT; preserved in [`LICENSE`](./LICENSE)
- LowRankArena exporter: [`export_lowrank_hf.py`](./export_lowrank_hf.py)
- Patch relative to the pinned upstream commit:
  [`LOWRANKARENA_UPSTREAM_DIFF.patch`](./LOWRANKARENA_UPSTREAM_DIFF.patch)

## LowRankArena changes

- Accept token-identical calibration tensors through
  `LOWRANKARENA_CALIBRATION_FILE`.
- Preserve token IDs above `uint16` when building calibration caches.
- Work around NumPy 2 dataset formatting during cache serialization.
- Export the resulting factors as an auditable Hugging Face low-rank artifact.

The primary upstream entrypoint remains [`main.py`](./main.py). The exact
LowRankArena sweep commands live at the repository level so paths, model IDs,
and shared calibration inputs remain explicit.

## Deliberate exclusions

Nested Git metadata, Python bytecode, logs, model weights, generated
checkpoints, caches, and run outputs are not part of the public snapshot.
