#!/bin/bash
set -euo pipefail

# run from repo root:
# bash baselines/SVD-LLM/train_SVD_LLM.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/baselines/SVD-LLM:${PYTHONPATH:-}"

CHECKPOINT_DIR=baselines/SVD-LLM/checkpoints
mkdir -p "${CHECKPOINT_DIR}"

# SVDLLM.py expects compression ratio as input and internally converts it to
# keep_ratio = 1 - ratio. Use 0.6 here to target a 40% keep ratio, matching
# the *_40 naming used by the other baselines.
COMPRESSION_RATIO=0.6
WHITENING_NSAMPLES=256
DATASET=wikitext2
SEED=3
SEQ_LEN=2048
STEP=1

# profiling matrices are huge for LLaMA models and are not needed for downstream eval,
# so the baseline scripts skip saving them by default.
DEVICE=cuda

GPU_7B=0
GPU_13B=0
GPU_30B=0

run_train() {
  local gpu="$1"
  local model_id="$2"
  local tag="$3"

  CUDA_VISIBLE_DEVICES="${gpu}" python baselines/SVD-LLM/SVDLLM.py \
    --model "${model_id}" \
    --step "${STEP}" \
    --ratio "${COMPRESSION_RATIO}" \
    --whitening_nsamples "${WHITENING_NSAMPLES}" \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --DEV "${DEVICE}" \
    --model_seq_len "${SEQ_LEN}" \
    --save_path "${CHECKPOINT_DIR}" \
    --timing_dir "${CHECKPOINT_DIR}" \
    --timing_file "${tag}_svdllm_timing.json" \
    --skip_profiling_save
}

# jeffwan llama-7b-hf, keep ratio ~= 0.4
run_train "${GPU_7B}" jeffwan/llama-7b-hf jeffwan_llama_7b_hf

# jeffwan llama-13b-hf, keep ratio ~= 0.4
run_train "${GPU_13B}" jeffwan/llama-13b-hf jeffwan_llama_13b_hf

# jeffwan llama-30b-hf, keep ratio ~= 0.4
run_train "${GPU_30B}" jeffwan/llama-30b-hf jeffwan_llama_30b_hf
