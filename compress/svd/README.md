# `compress/svd/`

This directory contains LowRankArena wrappers for low-rank artifact generation methods.

The files here are not intended to form one monolithic runtime. Their job is to present a uniform method surface to LowRankArena while preserving the fact that upstream low-rank baselines often have incompatible dependencies and export flows.

## Wrappers

- [`asvd.py`](./asvd.py): adapter for ASVD-style artifact generation.
- [`basis_sharing.py`](./basis_sharing.py): adapter for basis-sharing low-rank exports.
- [`dobi_svd.py`](./dobi_svd.py): adapter for Dobi-SVD export planning.
- [`fwsvd.py`](./fwsvd.py): placeholder wrapper for FWSVD integration.
- [`svd.py`](./svd.py): placeholder wrapper for plain SVD baselines.
- [`svd_llm.py`](./svd_llm.py): adapter for SVD-LLM export planning.

## Vendored Baselines

Several upstream low-rank projects are vendored or mirrored here as reference baselines:

- [`ASVD/`](./ASVD/README.md)
- [`Basis_Sharing/`](./Basis_Sharing/README.md)
- [`Dobi-SVD/`](./Dobi-SVD/readme.md)
- [`SVD-LLM/`](./SVD-LLM/README.md)

## Scope

- Build or plan low-rank artifacts.
- Emit uniform metadata through [`compress/save.py`](../save.py).
- Avoid owning evaluation or reporting logic.

## Runtime

Use the repo-level `compress` environment for low-rank artifact generation:

```bash
bash scripts/env/create_compress_env.sh
conda activate compress
```

SVD-LLM, Basis Sharing, MoDeGPT, Dobi-SVD, and ASVD compression/export imports are covered by this environment. ASVD's upstream direct eval path still expects the legacy `lm_eval.base` API, so run benchmark evaluation through LowRankArena instead.
