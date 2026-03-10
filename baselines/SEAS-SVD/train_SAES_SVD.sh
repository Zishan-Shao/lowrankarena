#!/bin/bash

# run from repo root:
# bash baselines/SEAS-SVD/train_SAES_SVD.sh

mkdir -p baselines/SEAS-SVD/robust

PARAM_RATIO=0.4
SEQ_LEN=2048
CALIB_SEQUENCES=128
BATCH_SIZE=1
MAX_TOKENS_TOTAL=262144
BETA_MODE=aces
SEED=42
DEVICE=cuda
TEACHER_DEVICE=cuda
DTYPE=bfloat16
TEACHER_DTYPE=bfloat16
FACTOR_DTYPE=bfloat16

GPU_7B=0
GPU_13B=0
GPU_30B=0

# jeffwan llama-7b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU_7B} python baselines/SEAS-SVD/saes_svd.py \
  --model_id jeffwan/llama-7b-hf \
  --output_dir baselines/SEAS-SVD/robust/jeffwan_llama-7b-hf_saes40 \
  --compression_ratio ${PARAM_RATIO} \
  --seq_len ${SEQ_LEN} \
  --calib_sequences ${CALIB_SEQUENCES} \
  --batch_size ${BATCH_SIZE} \
  --max_tokens_total ${MAX_TOKENS_TOTAL} \
  --beta_mode ${BETA_MODE} \
  --seed ${SEED} \
  --device ${DEVICE} \
  --teacher_device ${TEACHER_DEVICE} \
  --dtype ${DTYPE} \
  --teacher_dtype ${TEACHER_DTYPE} \
  --factor_dtype ${FACTOR_DTYPE}

# jeffwan llama-13b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU_13B} python baselines/SEAS-SVD/saes_svd.py \
  --model_id jeffwan/llama-13b-hf \
  --output_dir baselines/SEAS-SVD/robust/jeffwan_llama-13b-hf_saes40 \
  --compression_ratio ${PARAM_RATIO} \
  --seq_len ${SEQ_LEN} \
  --calib_sequences ${CALIB_SEQUENCES} \
  --batch_size ${BATCH_SIZE} \
  --max_tokens_total ${MAX_TOKENS_TOTAL} \
  --beta_mode ${BETA_MODE} \
  --seed ${SEED} \
  --device ${DEVICE} \
  --teacher_device ${TEACHER_DEVICE} \
  --dtype ${DTYPE} \
  --teacher_dtype ${TEACHER_DTYPE} \
  --factor_dtype ${FACTOR_DTYPE}

# jeffwan llama-30b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU_30B} python baselines/SEAS-SVD/saes_svd.py \
  --model_id jeffwan/llama-30b-hf \
  --output_dir baselines/SEAS-SVD/robust/jeffwan_llama-30b-hf_saes40 \
  --compression_ratio ${PARAM_RATIO} \
  --seq_len ${SEQ_LEN} \
  --calib_sequences ${CALIB_SEQUENCES} \
  --batch_size ${BATCH_SIZE} \
  --max_tokens_total ${MAX_TOKENS_TOTAL} \
  --beta_mode ${BETA_MODE} \
  --seed ${SEED} \
  --device ${DEVICE} \
  --teacher_device ${TEACHER_DEVICE} \
  --dtype ${DTYPE} \
  --teacher_dtype ${TEACHER_DTYPE} \
  --factor_dtype ${FACTOR_DTYPE}
