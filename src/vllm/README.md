# `src/vllm/`

This folder contains vLLM-specific adapter code and prototype utilities used to
make registered checkpoints loadable by vLLM.

## Why This Exists

Some low-rank Hugging Face artifacts are directly loadable by vLLM. Others need
a lightweight local wrapper so vLLM can resolve tokenizer mode, model class
registration, or dense compatibility shims. The wrapper generated here keeps
the original weights untouched and writes a loadable checkpoint directory for
the benchmark runner.

Materialized wrapper checkpoints default to an external cache under
`~/.cache/lowrankarena/vllm`. The tracked [`checkpoints/vllm/`](../../checkpoints/vllm/README.md)
folder is only for intentionally archived wrappers. Benchmark outputs should
live under [`results/`](../../results/README.md). `src/vllm/` itself is code-only.

The runtime scripts default to compact terminal output:

- suppress most vLLM internal logs
- disable checkpoint-load progress bars
- print short wrapper-owned stage updates

If you need raw vLLM logs for debugging, add `--verbose-vllm` on the prototype
scripts or `--verbose-backend` on the main runner.

Because this folder is named `vllm/`, the runtime entrypoints explicitly
resolve the installed vLLM package from the active environment so the local
folder name does not shadow the backend package.

## Files

- `prepare_svdllm_vllm_model.py`: build a local wrapper model directory from an existing low-rank checkpoint.
- `vllm_adapter.py`: decide whether direct vLLM loading is possible or whether a wrapper checkpoint is required.
- `prototype_speed_runner.py`: standalone prototype of the same flow used by the main speed runner.
- `test_vllm_svdllm.py`: run a minimal vLLM generation smoke test.
- `benchmark_vllm_svdllm.py`: run a simple latency/throughput benchmark on a wrapper model.

## Example

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lowrankarena

python src/vllm/prepare_svdllm_vllm_model.py \
  --source-model /path/to/anonymous/exported/checkpoint \
  --output-dir checkpoints/vllm/lowrank_demo_vllm

CUDA_VISIBLE_DEVICES=0 python src/vllm/test_vllm_svdllm.py \
  --model checkpoints/vllm/lowrank_demo_vllm

CUDA_VISIBLE_DEVICES=0 python src/vllm/benchmark_vllm_svdllm.py \
  --model checkpoints/vllm/lowrank_demo_vllm \
  --batch-size 1 \
  --prompt-length 512 \
  --generation-length 128

CUDA_VISIBLE_DEVICES=0 python src/vllm/prototype_speed_runner.py \
  --checkpoint-name llama31-8b-svdllm-0.6 \
  --batch-size 1 \
  --prompt-length 32 \
  --generation-length 8 \
  --repeat 1 \
  --warmup 0 \
  --gpu-memory-utilization 0.4 \
  --enforce-eager
```

The last command mirrors the preparation step used by
[`src/speed_runner.py`](../speed_runner.py).
