#!/bin/bash
set -euo pipefail

# run from repo root:
# bash baselines/SVD-LLM/train_SVD_LLMv2_smoke.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/baselines/SVD-LLM:${PYTHONPATH:-}"

CHECKPOINT_DIR=baselines/SVD-LLM/checkpoints_smoke
mkdir -p "${CHECKPOINT_DIR}"

COMPRESSION_RATIO=0.6
KEEP_RATIO=0.4
DATASET=wikitext2
NSAMPLES=8
SEQ_LEN=2048
BATCH_SIZE=1
SEED=3
LOAD_DTYPE=float16
STATS_DTYPE=float32
SQRT_DTYPE=float32
STORE_ACT_DTYPE=float16
GPU=6

# smoke build for jeffwan llama-7b-hf, keep ratio ~= 0.4
CUDA_VISIBLE_DEVICES="${GPU}" python -u baselines/SVD-LLM/SVDLLM_v2.py \
  --model jeffwan/llama-7b-hf \
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
  --save_path "${CHECKPOINT_DIR}/jeffwan_llama_7b_hf_svdllmv2_keep${KEEP_RATIO}_smoke.pt" \
  --timing_file jeffwan_llama_7b_hf_svdllmv2_smoke_timing.json \
  --tqdm auto
