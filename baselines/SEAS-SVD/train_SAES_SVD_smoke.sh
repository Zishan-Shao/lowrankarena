#!/bin/bash

# run from repo root:
# bash baselines/SEAS-SVD/train_SAES_SVD_smoke.sh

mkdir -p baselines/SEAS-SVD/robust

PARAM_RATIO=0.4
SEQ_LEN=2048
CALIB_SEQUENCES=8
BATCH_SIZE=1
MAX_TOKENS_TOTAL=16384
MAX_BATCHES=2
BETA_MODE=aces
SEED=42
DEVICE=cuda
TEACHER_DEVICE=cuda
DTYPE=bfloat16
TEACHER_DTYPE=bfloat16
FACTOR_DTYPE=bfloat16
GPU=3

# smoke build for jeffwan llama-7b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU} python baselines/SEAS-SVD/saes_svd.py \
  --model_id jeffwan/llama-7b-hf \
  --output_dir baselines/SEAS-SVD/robust/jeffwan_llama-7b-hf_saes40_smoke \
  --compression_ratio ${PARAM_RATIO} \
  --seq_len ${SEQ_LEN} \
  --calib_sequences ${CALIB_SEQUENCES} \
  --batch_size ${BATCH_SIZE} \
  --max_batches ${MAX_BATCHES} \
  --max_tokens_total ${MAX_TOKENS_TOTAL} \
  --beta_mode ${BETA_MODE} \
  --seed ${SEED} \
  --device ${DEVICE} \
  --teacher_device ${TEACHER_DEVICE} \
  --dtype ${DTYPE} \
  --teacher_dtype ${TEACHER_DTYPE} \
  --factor_dtype ${FACTOR_DTYPE}
