#!/bin/bash

# run from repo root:
# bash baselines/Dobi-SVD/train_dobi.sh
#
# these commands use the remapping version.
# if you want the non-remapping version, remove --remapping and update the folder names.

RESULTS_DIR=baselines/Dobi-SVD/results
TARGET_RATIO=0.4
SEQ_LEN=2048
SEED=0
TRAINING_DATASET=wikitext2
N_TRAIN_EPOCHS=20
N_TRAIN_SAMPLES=256
N_EVAL_SAMPLES=256

GPU_7B=0
GPU_13B=0
GPU_30B=0

# jeffwan llama-7b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU_7B} python baselines/Dobi-SVD/svd_trainer.py \
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

TRAINING_RESULT_PATH_7B=$(basename "$(ls -td ${RESULTS_DIR}/training_output/llama-7b-hf/Diff-Remapping-0.4_${TRAINING_DATASET}_${SEQ_LEN}_* | head -n 1)")
CUDA_VISIBLE_DEVICES=${GPU_7B} python baselines/Dobi-SVD/weight_updater.py \
  --model_id jeffwan/llama-7b-hf \
  --training_result_path ${TRAINING_RESULT_PATH_7B} \
  --seed ${SEED} \
  --n_train_samples ${N_TRAIN_SAMPLES} \
  --n_eval_samples ${N_EVAL_SAMPLES} \
  --training_dataset ${TRAINING_DATASET} \
  --path_head_folder baselines/Dobi-SVD \
  --path_head_folder_output ${RESULTS_DIR} \
  --remapping

# jeffwan llama-13b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU_13B} python baselines/Dobi-SVD/svd_trainer.py \
  --model_id jeffwan/llama-13b-hf \
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

TRAINING_RESULT_PATH_13B=$(basename "$(ls -td ${RESULTS_DIR}/training_output/llama-13b-hf/Diff-Remapping-0.4_${TRAINING_DATASET}_${SEQ_LEN}_* | head -n 1)")
CUDA_VISIBLE_DEVICES=${GPU_13B} python baselines/Dobi-SVD/weight_updater.py \
  --model_id jeffwan/llama-13b-hf \
  --training_result_path ${TRAINING_RESULT_PATH_13B} \
  --seed ${SEED} \
  --n_train_samples ${N_TRAIN_SAMPLES} \
  --n_eval_samples ${N_EVAL_SAMPLES} \
  --training_dataset ${TRAINING_DATASET} \
  --path_head_folder baselines/Dobi-SVD \
  --path_head_folder_output ${RESULTS_DIR} \
  --remapping

# jeffwan llama-30b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU_30B} python baselines/Dobi-SVD/svd_trainer.py \
  --model_id jeffwan/llama-30b-hf \
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

TRAINING_RESULT_PATH_30B=$(basename "$(ls -td ${RESULTS_DIR}/training_output/llama-30b-hf/Diff-Remapping-0.4_${TRAINING_DATASET}_${SEQ_LEN}_* | head -n 1)")
CUDA_VISIBLE_DEVICES=${GPU_30B} python baselines/Dobi-SVD/weight_updater.py \
  --model_id jeffwan/llama-30b-hf \
  --training_result_path ${TRAINING_RESULT_PATH_30B} \
  --seed ${SEED} \
  --n_train_samples ${N_TRAIN_SAMPLES} \
  --n_eval_samples ${N_EVAL_SAMPLES} \
  --training_dataset ${TRAINING_DATASET} \
  --path_head_folder baselines/Dobi-SVD \
  --path_head_folder_output ${RESULTS_DIR} \
  --remapping
