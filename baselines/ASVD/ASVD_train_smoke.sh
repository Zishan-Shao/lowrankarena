#!/bin/bash

# run from repo root:
# bash baselines/ASVD/ASVD_train_smoke.sh

PARAM_RATIO=0.4
ALPHA=0.5
CALIB_DATASET=wikitext2
N_CALIB_SAMPLES=8
CALIB_SEQLEN=2048
SEED=3
SCALING_METHOD=abs_mean
SENSITIVITY_METRIC=ppl
GPU=2

# smoke build for jeffwan llama-7b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU} python huggingface_repos/build_asvd_repo.py \
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
  --save_path huggingface_repos/jeffwan_llama-7b-hf_asvd40_smoke
