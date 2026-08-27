# AA-SVD: Anchored and Adaptive SVD for Large Language Model Compression

Official implementation of the paper: **AA-SVD: Anchored and Adaptive SVD for Large Language Model Compression**
[[arXiv]](https://arxiv.org/abs/2604.02119)

## Setup

Requires Python 3.10+ and a CUDA-capable GPU.

```bash
python -m venv venv  # or: virtualenv venv / conda create -n aa-svd python=3.10
source venv/bin/activate  # or: conda activate aa-svd
pip install -r requirements.txt
```

## Running Compression

```bash
python main.py model=llama-7B data=wikitext2 compression.target_param_ratio=0.8
```

`target_param_ratio` controls the fraction of parameters retained after compression (e.g. `0.8` retains 80%).

**Other examples:**

```bash
# Llama-2 13B on C4 calibration data, 70% parameter retention
python main.py model=llama2-13B data=c4 compression.target_param_ratio=0.7

# Qwen2.5-7B
python main.py model=qwen2.5-7B data=wikitext2 compression.target_param_ratio=0.8

# Evaluate dense model (no compression)
python main.py model=llama-7B compression.sub_method=no-compress
```

## Resuming and Evaluation

If compression was interrupted or has already been completed, you can resume or skip directly to evaluation by pointing `compression.save_path` at the previously saved weights:

```bash
python main.py compression.save_path=<path_to_saved_weights> model=llama-7B data=wikitext2
```

- **Partially compressed:** resumes compression from the saved checkpoint.
- **Fully compressed:** skips compression and runs evaluation on the saved weights.

> **Note:** The code does not explicitly check that the config matches the original run. A mismatch in model or compression settings may cause silent errors or a crash.

## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration. The top-level config is [config/config.yaml](config/config.yaml); it composes the following config groups:

| Group | Path | Purpose |
|---|---|---|
| `model` | `config/model/` | Model name, dtype, device map |
| `data` | `config/data/` | Calibration dataset and tokenization settings |
| `compression` | `config/compression/` | Compression method, rank allocation, finetuning |
| `paths` | `config/paths/` | `data_dir` (calibration data cache) and `output_dir` (logs and saved weights) |

Any field can be overridden on the command line:

```bash
python main.py model=llama3-8B compression.target_param_ratio=0.75 compression.num_calibration_samples=512
```

**Objective function:** Set `compression.sub_method` to select the layer-wise SVD objective as defined in the paper:

| Value | Description |
|---|---|
| `obj1` | Objective 1 |
| `obj2` | Objective 2 (default) |
| `obj3` | Objective 3 |
| `obj4` | Objective 4 |

```bash
python main.py model=llama-7B compression.sub_method=obj3 compression.target_param_ratio=0.8
```

**Block-level local refinement:** Enabled by default. To disable:

```bash
python main.py model=llama-7B compression.finetune.enabled=false compression.target_param_ratio=0.8
```

**Evaluation** tasks are configured under the `evaluate` key in [config/config.yaml](config/config.yaml). The default runs perplexity on WikiText-2/PTB/C4 and zero-shot accuracy on eight LM-Eval harness tasks. To add, remove, or change tasks edit the `evaluate.lm_eval.tasks` list.

## Paths

Default paths are defined in [config/paths/default.yaml](config/paths/default.yaml):

```yaml
data_dir: ${hydra:runtime.cwd}/data    # where preprocessed calibration data is cached
output_dir: ${hydra:runtime.cwd}/logs  # where logs and saved weights are written
```

These resolve relative to the project directory (the directory from which you run `main.py`). Both can be overridden on the command line:

```bash
python main.py paths.data_dir=/my/data paths.output_dir=/my/outputs \
  model=llama-7B data=wikitext2 compression.target_param_ratio=0.8
```

For cluster-specific path presets, add a new file under `config/paths/` and select it with `paths=<name>`.

## Logging with Weights & Biases

W&B logging is enabled by default. Make sure you are logged in (`wandb login`) and set `wandb.project` to your own project name. Configure it under the `wandb` key in [config/config.yaml](config/config.yaml):

```yaml
wandb:
  use: true       # set to false to disable
  project: aa-svd # replace with your W&B project name
  id: null        # set to a run ID to resume a specific run
  resume: null    # choices: null, must, allow, never
```

To disable W&B for a single run:

```bash
python main.py wandb.use=false model=llama-7B data=wikitext2 compression.target_param_ratio=0.8
```

## Extending the Framework

### Adding a new model

Model configs in `config/model/` use HuggingFace model IDs (e.g. `meta-llama/Llama-2-7b-hf`). Some models require accepting a license on HuggingFace and authenticating with `huggingface-cli login`. You can also substitute a local path for the `name` field.

1. Add a model config at `config/model/<model-name>.yaml`, following [config/model/llama-7B.yaml](config/model/llama-7B.yaml).
2. Implement a model adapter at `aa_svd/compression/adapters/<model>_adapter.py`, following [aa_svd/compression/adapters/llama_adapter.py](aa_svd/compression/adapters/llama_adapter.py). The adapter must subclass `ModelAdapter` and `LayerAdapter` from [aa_svd/compression/model_adapter.py](aa_svd/compression/model_adapter.py) and implement the required interface (layer iteration, embedding access, finetuning hooks, etc.).
3. Register the new adapter in [aa_svd/compression/adapters/\_\_init\_\_.py](aa_svd/compression/adapters/__init__.py) by importing the `ModelAdapter` subclass and adding an entry to `MODEL_ADAPTER_REGISTRY`.

### Adding a new calibration dataset

Add a data config at `config/data/<dataset-name>.yaml`, following [config/data/wikitext2.yaml](config/data/wikitext2.yaml).

### Changing evaluation

Edit the `evaluate` block in [config/config.yaml](config/config.yaml), or override individual fields directly on the command line:

```bash
# Add mmlu to lm_eval tasks and drop c4 from perplexity evaluation
python main.py \
  'evaluate.lm_eval.tasks=["winogrande","arc_challenge","piqa","mmlu"]' \
  'evaluate.ppl.datasets=["wikitext2","ptb"]'
```

## Citation

If you find our paper or this repository useful, please cite:

```bibtex
@article{sinha2026aasvd,
  title={AA-SVD: Anchored and Adaptive SVD for Large Language Model Compression},
  author={Sinha, Atul Kumar and Fleuret, Fran{\c{c}}ois},
  journal={arXiv preprint arXiv:2604.02119},
  year={2026}
}
```