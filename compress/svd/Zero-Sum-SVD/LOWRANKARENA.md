# ZS-SVD in LowRankArena

This directory is a source-only snapshot of
[Zero-Sum-SVD](https://github.com/mint-vu/Zero-Sum-SVD) with the calibration and
export changes used by LowRankArena. We use the short label **ZS-SVD** in tables
while retaining the upstream repository name in the directory path.

## Provenance

- Upstream commit: [`37e73f60875dbd5f0bf06327ae51d182c19fea33`](https://github.com/mint-vu/Zero-Sum-SVD/commit/37e73f60875dbd5f0bf06327ae51d182c19fea33)
- Upstream license: no standalone license file was present at the pinned commit
- LowRankArena exporter: [`export_lowrank_hf.py`](./export_lowrank_hf.py)
- Patch relative to the pinned upstream commit:
  [`LOWRANKARENA_UPSTREAM_DIFF.patch`](./LOWRANKARENA_UPSTREAM_DIFF.patch)

Because the upstream snapshot has no standalone license, this repository does
not relicense the upstream implementation under LowRankArena's MIT license.
Consult the upstream authors before reuse beyond inspection and reproducibility
review.

## LowRankArena changes

- Accept token-identical calibration tensors through
  `LOWRANKARENA_CALIBRATION_FILE`.
- Resolve pinned C4 files without copying datasets into the source tree.
- Key calibration caches by tokenizer identity and validate them on load.
- Export the one-shot zero-sum factors as an auditable Hugging Face artifact.

The primary source entrypoint is [`main_zero_sum.py`](./main_zero_sum.py), and
the LowRankArena export path is [`export_lowrank_hf.py`](./export_lowrank_hf.py).

## Deliberate exclusions

The bundled TextVQA dataset copy, diagnostic result figures, model/checkpoint
tensors, C4 copies, caches, logs, Python bytecode, and nested Git metadata are
not part of the public snapshot. Upstream branding assets referenced by the
upstream README are retained.
