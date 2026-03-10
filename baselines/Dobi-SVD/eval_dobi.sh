#!/bin/bash

# run from repo root:
# bash baselines/Dobi-SVD/eval_dobi.sh
#
# this script assumes the remapping version from train_dobi.sh.

mkdir -p outputs/dobi

LM_EVAL_TASKS=openbookqa,arc_easy,arc_challenge,winogrande,hellaswag,piqa,mathqa
PPL_DATASETS=wikitext2,ptb,c4
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

DOBI_MODEL_7B=baselines/Dobi-SVD/results/compressed_model/llama-7b-hf/DobiSVD-llama-7b-hf-0.4
DOBI_MODEL_13B=baselines/Dobi-SVD/results/compressed_model/llama-13b-hf/DobiSVD-llama-13b-hf-0.4
DOBI_MODEL_30B=baselines/Dobi-SVD/results/compressed_model/llama-30b-hf/DobiSVD-llama-30b-hf-0.4

# jeffwan llama-7b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU_BENCH_7B} python -m eval_results.eval_benchmarks \
  --dobi_model ${DOBI_MODEL_7B} \
  --dobi_remapping \
  --device ${DEVICE} \
  --batch_size ${BATCH_SIZE} \
  --use_lm_eval \
  --dtype ${DTYPE} \
  --lm_eval_tasks ${LM_EVAL_TASKS} \
  --output_json outputs/dobi/jeffwan_llama-7b-hf_dobi40_lm_eval.json

CUDA_VISIBLE_DEVICES=${GPU_PPL_7B} python -m eval_results.eval_general_ppl \
  --dobi_model ${DOBI_MODEL_7B} \
  --dobi_remapping \
  --datasets ${PPL_DATASETS} \
  --metrics token \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --output_json outputs/dobi/jeffwan_llama-7b-hf_dobi40_ppl.json

CUDA_VISIBLE_DEVICES=${GPU_LING_7B} python -m eval_results.eval_linguistic_tasks \
  --dobi_model ${DOBI_MODEL_7B} \
  --dobi_remapping \
  --tasks blimp \
  --device ${DEVICE} \
  --dtype ${LING_DTYPE} \
  --output_json outputs/dobi/jeffwan_llama-7b-hf_dobi40_ling.json

# jeffwan llama-13b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU_BENCH_13B} python -m eval_results.eval_benchmarks \
  --dobi_model ${DOBI_MODEL_13B} \
  --dobi_remapping \
  --device ${DEVICE} \
  --batch_size ${BATCH_SIZE} \
  --use_lm_eval \
  --dtype ${DTYPE} \
  --lm_eval_tasks ${LM_EVAL_TASKS} \
  --output_json outputs/dobi/jeffwan_llama-13b-hf_dobi40_lm_eval.json

CUDA_VISIBLE_DEVICES=${GPU_PPL_13B} python -m eval_results.eval_general_ppl \
  --dobi_model ${DOBI_MODEL_13B} \
  --dobi_remapping \
  --datasets ${PPL_DATASETS} \
  --metrics token \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --output_json outputs/dobi/jeffwan_llama-13b-hf_dobi40_ppl.json

CUDA_VISIBLE_DEVICES=${GPU_LING_13B} python -m eval_results.eval_linguistic_tasks \
  --dobi_model ${DOBI_MODEL_13B} \
  --dobi_remapping \
  --tasks blimp \
  --device ${DEVICE} \
  --dtype ${LING_DTYPE} \
  --output_json outputs/dobi/jeffwan_llama-13b-hf_dobi40_ling.json

# jeffwan llama-30b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU_BENCH_30B} python -m eval_results.eval_benchmarks \
  --dobi_model ${DOBI_MODEL_30B} \
  --dobi_remapping \
  --device ${DEVICE} \
  --batch_size ${BATCH_SIZE} \
  --use_lm_eval \
  --dtype ${DTYPE} \
  --lm_eval_tasks ${LM_EVAL_TASKS} \
  --output_json outputs/dobi/jeffwan_llama-30b-hf_dobi40_lm_eval.json

CUDA_VISIBLE_DEVICES=${GPU_PPL_30B} python -m eval_results.eval_general_ppl \
  --dobi_model ${DOBI_MODEL_30B} \
  --dobi_remapping \
  --datasets ${PPL_DATASETS} \
  --metrics token \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --output_json outputs/dobi/jeffwan_llama-30b-hf_dobi40_ppl.json

CUDA_VISIBLE_DEVICES=${GPU_LING_30B} python -m eval_results.eval_linguistic_tasks \
  --dobi_model ${DOBI_MODEL_30B} \
  --dobi_remapping \
  --tasks blimp \
  --device ${DEVICE} \
  --dtype ${LING_DTYPE} \
  --output_json outputs/dobi/jeffwan_llama-30b-hf_dobi40_ling.json
