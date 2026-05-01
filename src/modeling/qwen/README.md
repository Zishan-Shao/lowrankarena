# `src/modeling/qwen/`

Shared low-rank runtime for Qwen-family checkpoints.

This folder groups family-specific wrappers for supported Qwen lines while keeping the compression schema consistent across methods. Once a checkpoint is exported into the shared arena schema, the benchmark path should be able to load it through one of these family runtimes without method-specific forward code.

## Files

- [`configuration_lowrank_qwen2.py`](./configuration_lowrank_qwen2.py): Qwen2 config with arena low-rank metadata.
- [`modeling_lowrank_qwen2.py`](./modeling_lowrank_qwen2.py): Qwen2 runtime wrapper that installs shared low-rank linear layers.
- [`configuration_lowrank_qwen3.py`](./configuration_lowrank_qwen3.py): Qwen3 config with arena low-rank metadata.
- [`modeling_lowrank_qwen3.py`](./modeling_lowrank_qwen3.py): Qwen3 runtime wrapper that installs shared low-rank linear layers.

## Contract

- Keep this folder family-scoped rather than method-scoped.
- Treat these files as the remote-code boundary for exported Qwen-family arena artifacts.
- Keep export-time method logic outside this runtime, and keep vLLM preparation logic in [`src/vllm/`](../../vllm/README.md).
