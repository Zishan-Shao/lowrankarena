# `compress/prune/`

This directory contains LowRankArena wrappers for structured pruning baselines.

As with [`compress/svd/`](../svd/README.md), the unification target is the exported artifact contract rather than a shared pruning runtime.

## Wrappers

- [`bonsai.py`](./bonsai.py): adapter for Bonsai-style structured pruning.
- [`llm_pruner.py`](./llm_pruner.py): adapter for LLM-Pruner integration.
- [`slicegpt.py`](./slicegpt.py): adapter for SliceGPT-style structural compression.
- [`wanda_sp.py`](./wanda_sp.py): adapter for structured Wanda variants.

## Scope

- Resolve pruning-specific build requests.
- Preserve method identity and metadata in exported manifests.
- Keep external pruning repos optional and replaceable.
- Keep benchmark-time loading, eval, memory, and speed out of this directory.
