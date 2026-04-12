# `src/vllm/`

This folder contains the vLLM-specific adapter and prototype utilities used to make certain LowRankArena checkpoints loadable by vLLM.

## Why this exists

The original SVD-LLM HF artifact for `llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_then_update_keep50_hf` is close to usable in vLLM, but two compatibility issues block direct loading:

1. The fast tokenizer path is broken for this artifact, so vLLM must use `tokenizer_mode="slow"`.
2. vLLM's Transformers backend instantiates the model through `AutoModel.from_config(...)`, while the original checkpoint only exposes `AutoModelForCausalLM` in `auto_map`.

The wrapper generated here keeps the original weights untouched and only adds the missing base-model registration needed by vLLM.

Materialized wrapper checkpoints should live under [`checkpoints/vllm/`](../../checkpoints/vllm/README.md), and benchmark outputs should live under [`results/`](../../results/README.md).

The runtime scripts default to a compact terminal mode:

- suppress most vLLM internal logs
- disable checkpoint-load tqdm bars
- print short wrapper-owned stage updates instead

If you need raw vLLM logs for debugging, add `--verbose-vllm`.

Because this folder is named `vllm/`, the runtime entrypoints explicitly resolve the installed vLLM package from the conda environment so the local folder name does not shadow the backend package.

## Files

- `prepare_svdllm_vllm_model.py`: build a local wrapper model directory from an existing SVD-LLM checkpoint.
- `vllm_adapter.py`: prototype `prepare_model_for_vllm(...)` adapter that decides whether direct vLLM loading is possible.
- `prototype_speed_runner.py`: prototype `src/speed_runner.py` replacement that calls the adapter before `LLM(...)`.
- `test_vllm_svdllm.py`: run a minimal vLLM generation smoke test.
- `benchmark_vllm_svdllm.py`: run a simple latency/throughput benchmark on the wrapper model.

## Example

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lowrankarena

python /home/zs89/lowrankarena/src/vllm/prepare_svdllm_vllm_model.py \
  --source-model /home/zs89/.cache/huggingface/hub/models--Duke-CEI-SVD--LowRankArena/snapshots/6ce37f6e157cc9c689221fd1545a3c0c3c0efbf6/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_then_update_keep50_hf \
  --output-dir /home/zs89/lowrankarena/checkpoints/vllm/llama_7b_svdllm_keep50_vllm

CUDA_VISIBLE_DEVICES=0 python /home/zs89/lowrankarena/src/vllm/test_vllm_svdllm.py \
  --model /home/zs89/lowrankarena/checkpoints/vllm/llama_7b_svdllm_keep50_vllm

CUDA_VISIBLE_DEVICES=0 python /home/zs89/lowrankarena/src/vllm/benchmark_vllm_svdllm.py \
  --model /home/zs89/lowrankarena/checkpoints/vllm/llama_7b_svdllm_keep50_vllm \
  --batch-size 1 \
  --prompt-length 512 \
  --generation-length 128

CUDA_VISIBLE_DEVICES=0 python /home/zs89/lowrankarena/src/vllm/prototype_speed_runner.py \
  --checkpoint-name llama-7b-svdllm-v1-update-0.5 \
  --batch-size 1 \
  --prompt-length 32 \
  --generation-length 8 \
  --repeat 1 \
  --warmup 0 \
  --gpu-memory-utilization 0.4 \
  --enforce-eager
```

The last command is the important one for future LowRankArena integration: it mirrors the current `src/speed_runner.py` flow, but inserts `prepare_model_for_vllm(...)` between `load_checkpoint(...)` and `vllm.LLM(...)`.
