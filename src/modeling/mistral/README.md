# `src/modeling/mistral/`

Shared low-rank runtime for Mistral-family checkpoints.

This folder groups the arena-owned config and model wrappers for Mistral variants whose compressed linear layers have already been exported into the shared `ABLinear` schema. Method-specific preprocessing still belongs in [`compress/`](../../../compress/README.md); the files here are only the family-level runtime used after export.

## Files

- [`configuration_lowrank_mistral.py`](./configuration_lowrank_mistral.py): extends the upstream Mistral config with arena low-rank metadata.
- [`modeling_lowrank_mistral.py`](./modeling_lowrank_mistral.py): wraps the upstream Mistral model classes and replaces targeted linear modules with shared low-rank layers.

## Contract

- Keep this folder family-scoped rather than method-scoped.
- Treat these files as the remote-code boundary for exported Mistral-family arena artifacts.
- Keep export-time method logic in [`compress/`](../../../compress/README.md), and keep vLLM preparation logic in [`src/vllm/`](../../vllm/README.md).
