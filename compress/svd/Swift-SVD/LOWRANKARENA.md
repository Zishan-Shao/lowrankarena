# Swift-SVD in LowRankArena

This directory is a source-only snapshot of
[Swift-SVD](https://github.com/hiahei/Swift-SVD) with the compatibility and
export changes used by LowRankArena. The hundreds of gigabytes of local SVD
statistics and generated artifacts are deliberately excluded.

## Provenance

- Upstream commit: [`bd5be98340864deb5e51f120244ab43c446373d7`](https://github.com/hiahei/Swift-SVD/commit/bd5be98340864deb5e51f120244ab43c446373d7)
- Upstream license: MIT; preserved in [`LICENSE`](./LICENSE)
- LowRankArena exporter: [`export_lowrank_hf.py`](./export_lowrank_hf.py)
- Patch relative to the pinned upstream commit:
  [`LOWRANKARENA_UPSTREAM_DIFF.patch`](./LOWRANKARENA_UPSTREAM_DIFF.patch)

## LowRankArena changes

- Accept token-identical calibration tensors through
  `LOWRANKARENA_CALIBRATION_FILE`.
- Make WikiText-2 and pinned C4 cache discovery explicit and deterministic.
- Seed C4 sampling and stabilize legacy LLaMA tokenizer loading.
- Add uniform keep-ratio allocation and factorized Hugging Face export paths
  used by the matched LowRankArena protocol.

Relevant entrypoints are [`train_svd.py`](./train_svd.py),
[`uniform_rank_allocation.py`](./uniform_rank_allocation.py), and
[`export_lowrank_hf.py`](./export_lowrank_hf.py). Dynamic allocation remains in
the snapshot but is not substituted for the uniform-budget baseline.

## Deliberate exclusions

`svd_list/`, model/checkpoint tensors, calibration caches, generated models,
logs, result files, Python bytecode, and nested Git metadata are not part of the
public snapshot.
