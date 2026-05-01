# `src/modeling/llama/`

Shared low-rank runtime for Llama-family checkpoints.

These files define the arena-owned config and model wrappers for artifacts whose dense base model is a Llama variant and whose compressed layers have already been exported into the arena `ABLinear` schema. They are the family-level runtime that exported checkpoints rely on after preprocessing is finished.

## Files

- [`configuration_lowrank_llama.py`](./configuration_lowrank_llama.py): extends the upstream Llama config with low-rank metadata.
- [`modeling_lowrank_llama.py`](./modeling_lowrank_llama.py): wraps the upstream Llama model classes and replaces targeted linear modules with shared low-rank layers.

## Contract

- Exported checkpoints should already be in the shared arena low-rank schema before they depend on this runtime.
- These files are part of the remote-code boundary copied into loadable Hugging Face artifacts.
- Method-specific preprocessing and vLLM compatibility preparation do not belong here. vLLM preparation lives in [`src/vllm/`](../../vllm/README.md).
