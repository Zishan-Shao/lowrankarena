# `src/modeling/`

This directory contains the shared low-rank runtime used by registered arena artifacts.

The key design choice is that runtime support is grouped by supported base-model family, not by compression method. Method-specific code belongs in [`compress/`](../../compress/README.md); once an artifact is exported into the arena low-rank schema, execution should route through one of the shared family runtimes here.

## Layout

- [`common.py`](./common.py): shared low-rank layer primitives and module-replacement helpers.
- [`llama/`](./llama/README.md): Llama-family low-rank configs and model wrappers.
- [`qwen/`](./qwen/README.md): Qwen-family low-rank configs and model wrappers.

## Contract

- Base architecture must belong to a supported family.
- Compressed linear layers must be exported in the arena `ABLinear` schema.
- Family-specific runtime files are copied into exported Hugging Face artifacts as the remote-code boundary.
