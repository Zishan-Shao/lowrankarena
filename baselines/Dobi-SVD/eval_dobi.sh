#!/bin/bash
set -euo pipefail

# run from repo root:
# bash baselines/Dobi-SVD/eval_dobi.sh
#
# This evaluates the UNREMAPPING export.

mkdir -p outputs/dobi

MODEL_7B=baselines/Dobi-SVD/results/compressed_model/llama-7b-hf/DobiSVD_Noremapping-llama-7b-hf-0.4
MODEL_13B=baselines/Dobi-SVD/results/compressed_model/llama-13b-hf/DobiSVD_Noremapping-llama-13b-hf-0.4
MODEL_30B=baselines/Dobi-SVD/results/compressed_model/llama-30b-hf/DobiSVD_Noremapping-llama-30b-hf-0.4

TASKS_MAIN=openbookqa,arc_easy,arc_challenge,winogrande,hellaswag,piqa
TASKS_LING=blimp
DEVICE=cuda
DTYPE=bfloat16
BATCH_SIZE=1

GPU_7B=0
GPU_13B=0
GPU_30B=0

SVDLLM_TOKENIZER_MODEL=${MODEL_7B} CUDA_VISIBLE_DEVICES=${GPU_7B} python -m eval_results.eval_benchmarks \
  --dobi_model ${MODEL_7B} \
  --tokenizer ${MODEL_7B} \
  --device ${DEVICE} \
  --batch_size ${BATCH_SIZE} \
  --use_lm_eval \
  --dtype ${DTYPE} \
  --lm_eval_tasks ${TASKS_MAIN} \
  --output_json outputs/dobi/jeffwan_llama-7b-hf_dobi40_lm_eval.json

SVDLLM_TOKENIZER_MODEL=${MODEL_7B} CUDA_VISIBLE_DEVICES=${GPU_7B} python -m eval_results.eval_linguistic_tasks \
  --dobi_model ${MODEL_7B} \
  --tokenizer ${MODEL_7B} \
  --tasks ${TASKS_LING} \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --output_json outputs/dobi/jeffwan_llama-7b-hf_dobi40_ling.json

SVDLLM_TOKENIZER_MODEL=${MODEL_13B} CUDA_VISIBLE_DEVICES=${GPU_13B} python -m eval_results.eval_benchmarks \
  --dobi_model ${MODEL_13B} \
  --tokenizer ${MODEL_13B} \
  --device ${DEVICE} \
  --batch_size ${BATCH_SIZE} \
  --use_lm_eval \
  --dtype ${DTYPE} \
  --lm_eval_tasks ${TASKS_MAIN} \
  --output_json outputs/dobi/jeffwan_llama-13b-hf_dobi40_lm_eval.json

SVDLLM_TOKENIZER_MODEL=${MODEL_13B} CUDA_VISIBLE_DEVICES=${GPU_13B} python -m eval_results.eval_linguistic_tasks \
  --dobi_model ${MODEL_13B} \
  --tokenizer ${MODEL_13B} \
  --tasks ${TASKS_LING} \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --output_json outputs/dobi/jeffwan_llama-13b-hf_dobi40_ling.json

SVDLLM_TOKENIZER_MODEL=${MODEL_30B} CUDA_VISIBLE_DEVICES=${GPU_30B} python -m eval_results.eval_benchmarks \
  --dobi_model ${MODEL_30B} \
  --tokenizer ${MODEL_30B} \
  --device ${DEVICE} \
  --batch_size ${BATCH_SIZE} \
  --use_lm_eval \
  --dtype ${DTYPE} \
  --lm_eval_tasks ${TASKS_MAIN} \
  --output_json outputs/dobi/jeffwan_llama-30b-hf_dobi40_lm_eval.json

SVDLLM_TOKENIZER_MODEL=${MODEL_30B} CUDA_VISIBLE_DEVICES=${GPU_30B} python -m eval_results.eval_linguistic_tasks \
  --dobi_model ${MODEL_30B} \
  --tokenizer ${MODEL_30B} \
  --tasks ${TASKS_LING} \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --output_json outputs/dobi/jeffwan_llama-30b-hf_dobi40_ling.json
