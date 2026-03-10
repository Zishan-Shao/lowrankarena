#!/bin/bash

# run from repo root:
# bash baselines/ASVD/ASVD_train.sh

PARAM_RATIO=0.4
ALPHA=0.5
CALIB_DATASET=wikitext2
N_CALIB_SAMPLES=256
CALIB_SEQLEN=2048
SEED=3
SCALING_METHOD=abs_mean
SENSITIVITY_METRIC=ppl

GPU_7B=0
GPU_13B=0
GPU_30B=0

# jeffwan llama-7b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU_7B} python baselines/ASVD/huggingface_repos/build_asvd_repo.py \
  --model_id jeffwan/llama-7b-hf \
  --act_aware \
  --alpha ${ALPHA} \
  --n_calib_samples ${N_CALIB_SAMPLES} \
  --calib_dataset ${CALIB_DATASET} \
  --calib_seqlen ${CALIB_SEQLEN} \
  --seed ${SEED} \
  --scaling_method ${SCALING_METHOD} \
  --sensitivity_metric ${SENSITIVITY_METRIC} \
  --param_ratio_target ${PARAM_RATIO} \
  --save_path baselines/ASVD/huggingface_repos/jeffwan_llama-7b-hf_asvd40

# jeffwan llama-13b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU_13B} python baselines/ASVD/huggingface_repos/build_asvd_repo.py \
  --model_id jeffwan/llama-13b-hf \
  --act_aware \
  --alpha ${ALPHA} \
  --n_calib_samples ${N_CALIB_SAMPLES} \
  --calib_dataset ${CALIB_DATASET} \
  --calib_seqlen ${CALIB_SEQLEN} \
  --seed ${SEED} \
  --scaling_method ${SCALING_METHOD} \
  --sensitivity_metric ${SENSITIVITY_METRIC} \
  --param_ratio_target ${PARAM_RATIO} \
  --save_path baselines/ASVD/huggingface_repos/jeffwan_llama-13b-hf_asvd40

# jeffwan llama-30b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU_30B} python baselines/ASVD/huggingface_repos/build_asvd_repo.py \
  --model_id jeffwan/llama-30b-hf \
  --act_aware \
  --alpha ${ALPHA} \
  --n_calib_samples ${N_CALIB_SAMPLES} \
  --calib_dataset ${CALIB_DATASET} \
  --calib_seqlen ${CALIB_SEQLEN} \
  --seed ${SEED} \
  --scaling_method ${SCALING_METHOD} \
  --sensitivity_metric ${SENSITIVITY_METRIC} \
  --param_ratio_target ${PARAM_RATIO} \
  --save_path baselines/ASVD/huggingface_repos/jeffwan_llama-30b-hf_asvd40
