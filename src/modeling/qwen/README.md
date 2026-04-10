# `src/modeling/qwen/`

Shared low-rank runtime for Qwen-family checkpoints.

This folder groups family-specific wrappers for supported Qwen lines while keeping the compression schema consistent across methods.

## Files

- [`configuration_lowrank_qwen2.py`](./configuration_lowrank_qwen2.py): Qwen2 config with arena low-rank metadata.
- [`modeling_lowrank_qwen2.py`](./modeling_lowrank_qwen2.py): Qwen2 runtime wrapper that installs shared low-rank linear layers.
- [`configuration_lowrank_qwen3.py`](./configuration_lowrank_qwen3.py): Qwen3 config with arena low-rank metadata.
- [`modeling_lowrank_qwen3.py`](./modeling_lowrank_qwen3.py): Qwen3 runtime wrapper that installs shared low-rank linear layers.

This folder is intentionally family-scoped. New compression methods should export into this runtime rather than ship per-checkpoint forward code.
