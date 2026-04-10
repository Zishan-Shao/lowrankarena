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

## Scope

- Build or plan low-rank artifacts.
- Emit uniform metadata through [`compress/save.py`](../save.py).
- Avoid owning evaluation or reporting logic.
