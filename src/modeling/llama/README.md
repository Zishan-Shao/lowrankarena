# `src/modeling/llama/`

Shared low-rank runtime for Llama-family checkpoints.

These files define the arena-owned config and model wrappers for artifacts whose dense base model is a Llama variant and whose compressed layers have already been exported into the arena `ABLinear` schema.

## Files

- [`configuration_lowrank_llama.py`](./configuration_lowrank_llama.py): extends the upstream Llama config with low-rank metadata.
- [`modeling_lowrank_llama.py`](./modeling_lowrank_llama.py): wraps the upstream Llama model classes and replaces targeted linear modules with shared low-rank layers.

Method-specific preprocessing does not belong here. By the time a checkpoint reaches this folder's runtime, the artifact should already be benchmark-ready.
