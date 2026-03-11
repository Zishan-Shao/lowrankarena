#!/bin/bash
set -euo pipefail

# run from repo root:
# bash baselines/SVD-LLM/train_SVD_LLMv2.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/baselines/SVD-LLM:${PYTHONPATH:-}"

CHECKPOINT_DIR=baselines/SVD-LLM/checkpoints
mkdir -p "${CHECKPOINT_DIR}"

# SVDLLM_v2 exposes ratio_type explicitly. Use reduction=0.6 to target a 40%
# keep ratio so the runs line up with the other 40%-parameter baselines.
COMPRESSION_RATIO=0.6
KEEP_RATIO=0.4
DATASET=wikitext2
NSAMPLES=256
SEQ_LEN=2048
BATCH_SIZE=1
SEED=3
LOAD_DTYPE=float16
STATS_DTYPE=float32
SQRT_DTYPE=float32
STORE_ACT_DTYPE=float16

GPU_7B=0
GPU_13B=0
GPU_30B=0

run_train() {
  local gpu="$1"
  local model_id="$2"
  local tag="$3"

  CUDA_VISIBLE_DEVICES="${gpu}" python -u baselines/SVD-LLM/SVDLLM_v2.py \
    --model "${model_id}" \
    --dataset "${DATASET}" \
    --nsamples "${NSAMPLES}" \
    --seq_len "${SEQ_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --ratio "${COMPRESSION_RATIO}" \
    --ratio_type reduction \
    --device cuda:0 \
    --seed "${SEED}" \
    --load_dtype "${LOAD_DTYPE}" \
    --stats_dtype "${STATS_DTYPE}" \
    --sqrt_dtype "${SQRT_DTYPE}" \
    --store_act_dtype "${STORE_ACT_DTYPE}" \
    --save_path "${CHECKPOINT_DIR}/${tag}_svdllmv2_keep${KEEP_RATIO}.pt" \
    --timing_file "${tag}_svdllmv2_timing.json" \
    --tqdm auto
}

# jeffwan llama-7b-hf, keep ratio ~= 0.4
run_train "${GPU_7B}" jeffwan/llama-7b-hf jeffwan_llama_7b_hf

# jeffwan llama-13b-hf, keep ratio ~= 0.4
run_train "${GPU_13B}" jeffwan/llama-13b-hf jeffwan_llama_13b_hf

# jeffwan llama-30b-hf, keep ratio ~= 0.4
run_train "${GPU_30B}" jeffwan/llama-30b-hf jeffwan_llama_30b_hf
