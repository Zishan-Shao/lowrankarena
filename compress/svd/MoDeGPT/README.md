# A code implemention of MoDeGPT

This repository is a code implemention of [MoDeGPT](https://arxiv.org/abs/2408.09632) in ICLR 2025

### 1. Set-up

1. Clone this repository

```bash
git clone https://github.com/cbacary/MoDeGPT.git
cd MoDeGPT
```

2. Install Package

using `uv`:

```Shell

uv venv --python 3.12.6
uv sync
```

### 2. Run compression

See tests.sh for run example. See src/adapters/CompressionConfig.py for a full list of cli args.
Each arg in CompressionConfig directly translates to a cli arg of the form `--dataset wikitext`

Must do:

```bash
python -m src.run_modegpt [options]
```

## Useful options

See `tests.sh` for useful examples on running Qwen models. You should simply be able to swap out the MODEL_NAME with a different supported model and be able to run.

`--temp_storage_dir` - a directory to output the compressed versions of the layers. Ideally, point this to a location with fast read/write speeds.

### 3. Additional Information

Currently tested against OPT, Llama2-7b, and llama3-8b models.

To implement for additional model architectures you will have to modify / create your own patched version of modeling_llama (different for each architecture). See `patchers/OPTRebuild.py`, `patchers/LlamaRebuild.py` and the `patch_config` function in `patchers/patch.py`. This process mostly involves changing the dimensions of the linear layers when they are initialized to the compressed dimensions, but other steps may be required depending on the architecture you are working with. For example, llama3-8b requires handling RoPE.

Additionally, you will also have to implement a custom adapter for the model which implements the interface defined in model_adapter.
