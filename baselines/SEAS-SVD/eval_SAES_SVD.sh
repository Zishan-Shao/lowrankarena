#!/bin/bash

# run from repo root:
# bash baselines/SEAS-SVD/eval_SAES_SVD.sh

mkdir -p outputs/saes

LM_EVAL_TASKS=openbookqa,arc_easy,arc_challenge,winogrande,hellaswag,piqa,mathqa,truthfulqa_mc1
PPL_DATASETS=wikitext2,ptb,c4
SEQ_LEN=2048
C4_DOCS=2000
BATCH_SIZE=1
DEVICE=cuda
DTYPE=bfloat16
LING_DTYPE=bfloat16

GPU_BENCH_7B=0
GPU_PPL_7B=0
GPU_LING_7B=0

GPU_BENCH_13B=0
GPU_PPL_13B=0
GPU_LING_13B=0

GPU_BENCH_30B=0
GPU_PPL_30B=0
GPU_LING_30B=0

SAES_MODEL_7B=baselines/SEAS-SVD/robust/jeffwan_llama-7b-hf_saes40
SAES_MODEL_13B=baselines/SEAS-SVD/robust/jeffwan_llama-13b-hf_saes40
SAES_MODEL_30B=baselines/SEAS-SVD/robust/jeffwan_llama-30b-hf_saes40

# jeffwan llama-7b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU_BENCH_7B} python -m eval_results.eval_benchmarks \
  --model ${SAES_MODEL_7B} \
  --saes_svd \
  --saes_base_model jeffwan/llama-7b-hf \
  --use_lm_eval \
  --lm_eval_tasks ${LM_EVAL_TASKS} \
  --lm_eval_num_fewshot 0 \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --batch_size ${BATCH_SIZE} \
  --output_json outputs/saes/jeffwan_llama-7b-hf_saes40_lm_eval.json

CUDA_VISIBLE_DEVICES=${GPU_PPL_7B} python -m eval_results.eval_general_ppl \
  --saes_model ${SAES_MODEL_7B} \
  --saes_base_model jeffwan/llama-7b-hf \
  --datasets ${PPL_DATASETS} \
  --c4_stream \
  --c4_docs ${C4_DOCS} \
  --seqlen ${SEQ_LEN} \
  --batch_size ${BATCH_SIZE} \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --metrics token \
  --output_json outputs/saes/jeffwan_llama-7b-hf_saes40_ppl.json

CUDA_VISIBLE_DEVICES=${GPU_LING_7B} python -m eval_results.eval_linguistic_tasks \
  --saes_model ${SAES_MODEL_7B} \
  --saes_base_model jeffwan/llama-7b-hf \
  --tasks blimp \
  --num_fewshot 0 \
  --batch_size ${BATCH_SIZE} \
  --device ${DEVICE} \
  --dtype ${LING_DTYPE} \
  --output_json outputs/saes/jeffwan_llama-7b-hf_saes40_ling.json

# jeffwan llama-13b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU_BENCH_13B} python -m eval_results.eval_benchmarks \
  --model ${SAES_MODEL_13B} \
  --saes_svd \
  --saes_base_model jeffwan/llama-13b-hf \
  --use_lm_eval \
  --lm_eval_tasks ${LM_EVAL_TASKS} \
  --lm_eval_num_fewshot 0 \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --batch_size ${BATCH_SIZE} \
  --output_json outputs/saes/jeffwan_llama-13b-hf_saes40_lm_eval.json

CUDA_VISIBLE_DEVICES=${GPU_PPL_13B} python -m eval_results.eval_general_ppl \
  --saes_model ${SAES_MODEL_13B} \
  --saes_base_model jeffwan/llama-13b-hf \
  --datasets ${PPL_DATASETS} \
  --c4_stream \
  --c4_docs ${C4_DOCS} \
  --seqlen ${SEQ_LEN} \
  --batch_size ${BATCH_SIZE} \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --metrics token \
  --output_json outputs/saes/jeffwan_llama-13b-hf_saes40_ppl.json

CUDA_VISIBLE_DEVICES=${GPU_LING_13B} python -m eval_results.eval_linguistic_tasks \
  --saes_model ${SAES_MODEL_13B} \
  --saes_base_model jeffwan/llama-13b-hf \
  --tasks blimp \
  --num_fewshot 0 \
  --batch_size ${BATCH_SIZE} \
  --device ${DEVICE} \
  --dtype ${LING_DTYPE} \
  --output_json outputs/saes/jeffwan_llama-13b-hf_saes40_ling.json

# jeffwan llama-30b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU_BENCH_30B} python -m eval_results.eval_benchmarks \
  --model ${SAES_MODEL_30B} \
  --saes_svd \
  --saes_base_model jeffwan/llama-30b-hf \
  --use_lm_eval \
  --lm_eval_tasks ${LM_EVAL_TASKS} \
  --lm_eval_num_fewshot 0 \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --batch_size ${BATCH_SIZE} \
  --output_json outputs/saes/jeffwan_llama-30b-hf_saes40_lm_eval.json

CUDA_VISIBLE_DEVICES=${GPU_PPL_30B} python -m eval_results.eval_general_ppl \
  --saes_model ${SAES_MODEL_30B} \
  --saes_base_model jeffwan/llama-30b-hf \
  --datasets ${PPL_DATASETS} \
  --c4_stream \
  --c4_docs ${C4_DOCS} \
  --seqlen ${SEQ_LEN} \
  --batch_size ${BATCH_SIZE} \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --metrics token \
  --output_json outputs/saes/jeffwan_llama-30b-hf_saes40_ppl.json

CUDA_VISIBLE_DEVICES=${GPU_LING_30B} python -m eval_results.eval_linguistic_tasks \
  --saes_model ${SAES_MODEL_30B} \
  --saes_base_model jeffwan/llama-30b-hf \
  --tasks blimp \
  --num_fewshot 0 \
  --batch_size ${BATCH_SIZE} \
  --device ${DEVICE} \
  --dtype ${LING_DTYPE} \
  --output_json outputs/saes/jeffwan_llama-30b-hf_saes40_ling.json
