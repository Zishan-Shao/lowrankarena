#!/bin/bash
set -euo pipefail

# run from repo root:
# bash baselines/SVD-LLM/train_SVD_LLM_smoke.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/baselines/SVD-LLM:${PYTHONPATH:-}"

CHECKPOINT_DIR=baselines/SVD-LLM/checkpoints_smoke
mkdir -p "${CHECKPOINT_DIR}"

COMPRESSION_RATIO=0.6
WHITENING_NSAMPLES=8
DATASET=wikitext2
SEED=3
SEQ_LEN=2048
STEP=1

# profiling matrices are huge for LLaMA models and are not needed for downstream eval,
# so the baseline scripts skip saving them by default.
DEVICE=cuda
GPU=6

# smoke build for jeffwan llama-7b-hf, keep ratio ~= 0.4
CUDA_VISIBLE_DEVICES="${GPU}" python baselines/SVD-LLM/SVDLLM.py \
  --model jeffwan/llama-7b-hf \
  --step "${STEP}" \
  --ratio "${COMPRESSION_RATIO}" \
  --whitening_nsamples "${WHITENING_NSAMPLES}" \
  --dataset "${DATASET}" \
  --seed "${SEED}" \
  --DEV "${DEVICE}" \
  --model_seq_len "${SEQ_LEN}" \
  --save_path "${CHECKPOINT_DIR}" \
  --timing_dir "${CHECKPOINT_DIR}" \
  --timing_file jeffwan_llama_7b_hf_svdllm_smoke_timing.json \
  --skip_profiling_save
