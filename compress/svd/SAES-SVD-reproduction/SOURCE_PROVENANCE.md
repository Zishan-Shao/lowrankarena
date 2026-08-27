# Source provenance

The release files were extracted on 2026-08-27 from the local
`FlashSVDTrain` research workspace associated with
`https://github.com/Zishan-Shao/FlashSVDTrain.git`.

- Workspace base commit: `2681d0a75a6e95cc440d300e5ca2b013ade87f94`
- The `numerical_experiment/robust/saes_svd/` source directory was untracked in
  that workspace and is therefore not represented by the base commit alone.
- The source snapshot preserves the paper-aligned `saes_svd.py` implementation
  and its evaluation code. Checkpoints, caches, datasets, logs, bytecode, and
  unrelated FlashSVDTrain experiments were deliberately excluded.
- Only cosmetic module-docstring paths and comments were changed to point to
  the public release location. The commands in this release's
  [`README.md`](./README.md) are the canonical public entrypoints.

`SOURCE_SHA256SUMS` records hashes of the released Python source files after
extraction and those cosmetic release-path changes. This provenance record does
not claim that the source is an official implementation from the SAES-SVD paper
authors.
