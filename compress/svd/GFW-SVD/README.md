# GFW-SVD source snapshot

This directory contains a curated, source-only snapshot associated with
[Generalized Fisher-Weighted SVD (GFW-SVD)](https://arxiv.org/abs/2505.17974).
It is provided for source inspection and reproducibility work while the
LowRankArena integration is being finalized.

GFW-SVD is distinct from the older FWSVD baseline already present in
[`../FWSVD/`](../FWSVD/).

## Provenance

- Upstream repository:
  [`sayankotor/FisherKronecker`](https://github.com/sayankotor/FisherKronecker)
- Upstream commit:
  [`d009b028c1e73545d8c604bcd29c1e091c8f341c`](https://github.com/sayankotor/FisherKronecker/commit/d009b028c1e73545d8c604bcd29c1e091c8f341c)
- Paper: [arXiv:2505.17974](https://arxiv.org/abs/2505.17974)
- Original citation notice: [`UPSTREAM_README.md`](./UPSTREAM_README.md)
- Upstream license: no standalone license file was present at the pinned commit

Because the upstream snapshot has no standalone license, this repository does
not relicense the upstream implementation under LowRankArena's MIT license.
Consult the upstream authors before reuse beyond inspection and reproducibility
review.

## Included source

The `llama/` directory preserves the reusable LLaMA calibration, compression,
and layer-selection scripts from the pinned upstream commit:

- [`calibrate_llama_with_kronsvd.py`](./llama/calibrate_llama_with_kronsvd.py)
- [`compress_llama_with_kronsvd.py`](./llama/compress_llama_with_kronsvd.py)
- [`compress_llama_with_kronsvd_llama2.py`](./llama/compress_llama_with_kronsvd_llama2.py)
- [`llama_chat_select_layers.py`](./llama/llama_chat_select_layers.py)

The calibration example is preserved in [`llama/calib.sh`](./llama/calib.sh).

## Current limitations

This is an audit-oriented source snapshot, not yet a turnkey LowRankArena
artifact generator. The upstream compression and layer-selection scripts
retain author-environment paths and expect precomputed Kronecker-factor and
sensitivity artifacts. Replace those paths and supply the required artifacts
before running them. We do not claim that this snapshot alone reproduces the
paper's reported results.

## LowRankArena unified adapter

Do not run the hard-coded upstream `__main__` blocks. LowRankArena exposes the
pinned numerical helpers through [`../gfw_svd.py`](../gfw_svd.py):

```bash
python scripts/run_compress.py \
  --family svd \
  --method gfw_svd \
  --model meta-llama/Llama-3.1-8B \
  --ratio 0.5 \
  --extra kron_factors_dir=/path/to/fisher_factors \
  --preflight-only
```

After preflight succeeds, replace `--preflight-only` with `--execute`. The
adapter parameterizes every path, validates all required projection factors
before factorization, and writes a loadable LowRankArena ABLinear Hugging Face
artifact. An optional `--extra rank_config=/path/to/ranks.json` can provide
per-module keep ratios; otherwise the requested ratio is uniform.

## Deliberate exclusions

Nested Git metadata, notebooks, logs, generated figures, model/checkpoint
tensors, Kronecker factors, sensitivity outputs, and run results are not part
of this snapshot. One upstream file named `full_layers.py` contained captured
runtime log output rather than Python source and is also omitted.
