#!/bin/bash

# run from repo root:
# bash baselines/Dobi-SVD/train_dobi_smoke.sh
#
# smoke version for jeffwan llama-7b-hf, 40% params.
# this uses a separate output root so it does not overwrite the full run.

RESULTS_DIR=baselines/Dobi-SVD/results_smoke
TARGET_RATIO=0.4
SEQ_LEN=2048
SEED=0
TRAINING_DATASET=wikitext2
N_TRAIN_EPOCHS=1
N_TRAIN_SAMPLES=8
N_EVAL_SAMPLES=8
GPU=5

CUDA_VISIBLE_DEVICES=${GPU} python baselines/Dobi-SVD/svd_trainer.py \
  --model_id jeffwan/llama-7b-hf \
  --target_ratio ${TARGET_RATIO} \
  --seq_len ${SEQ_LEN} \
  --seed ${SEED} \
  --training_dataset ${TRAINING_DATASET} \
  --n_train_epochs ${N_TRAIN_EPOCHS} \
  --n_train_samples ${N_TRAIN_SAMPLES} \
  --n_eval_samples ${N_EVAL_SAMPLES} \
  --path_head_folder baselines/Dobi-SVD \
  --path_head_folder_output ${RESULTS_DIR} \
  --remapping

TRAINING_RESULT_PATH=$(basename "$(ls -td ${RESULTS_DIR}/training_output/llama-7b-hf/Diff-Remapping-0.4_${TRAINING_DATASET}_${SEQ_LEN}_* | head -n 1)")
CUDA_VISIBLE_DEVICES=${GPU} python baselines/Dobi-SVD/weight_updater.py \
  --model_id jeffwan/llama-7b-hf \
  --training_result_path ${TRAINING_RESULT_PATH} \
  --seed ${SEED} \
  --n_train_samples ${N_TRAIN_SAMPLES} \
  --n_eval_samples ${N_EVAL_SAMPLES} \
  --training_dataset ${TRAINING_DATASET} \
  --path_head_folder baselines/Dobi-SVD \
  --path_head_folder_output ${RESULTS_DIR} \
  --remapping
