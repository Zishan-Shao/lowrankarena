#!/bin/bash

# run from repo root:
# bash baselines/ASVD/ASVD_eval.sh

mkdir -p outputs/asvd

LM_EVAL_TASKS=openbookqa,arc_easy,arc_challenge,winogrande,hellaswag,piqa,mathqa,truthfulqa_mc1
PPL_DATASETS=wikitext2,ptb,c4
BATCH_SIZE=1
DEVICE=cuda
DTYPE=bfloat16
LING_DTYPE=fp16

GPU_BENCH_7B=0
GPU_PPL_7B=0
GPU_LING_7B=0

GPU_BENCH_13B=0
GPU_PPL_13B=0
GPU_LING_13B=0

GPU_BENCH_30B=0
GPU_PPL_30B=0
GPU_LING_30B=0

ASVD_MODEL_7B=baselines/ASVD/huggingface_repos/jeffwan_llama-7b-hf_asvd40
ASVD_MODEL_13B=baselines/ASVD/huggingface_repos/jeffwan_llama-13b-hf_asvd40
ASVD_MODEL_30B=baselines/ASVD/huggingface_repos/jeffwan_llama-30b-hf_asvd40

# jeffwan llama-7b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU_BENCH_7B} python -m eval_results.eval_benchmarks \
  --model ${ASVD_MODEL_7B} \
  --use_lm_eval \
  --lm_eval_tasks ${LM_EVAL_TASKS} \
  --lm_eval_num_fewshot 0 \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --batch_size ${BATCH_SIZE} \
  --output_json outputs/asvd/jeffwan_llama-7b-hf_asvd40_lm_eval.json

CUDA_VISIBLE_DEVICES=${GPU_PPL_7B} python -m baselines.ASVD.eval_ASVD_ppl_with_json \
  --checkpoint ${ASVD_MODEL_7B} \
  --datasets ${PPL_DATASETS} \
  --device ${DEVICE} \
  --seqlen 2048 \
  --batch_size ${BATCH_SIZE} \
  --dtype ${DTYPE} \
  --output_json outputs/asvd/jeffwan_llama-7b-hf_asvd40_ppl.json

CUDA_VISIBLE_DEVICES=${GPU_LING_7B} python -m baselines.ASVD.eval_ASVD_linguistic_tasks_with_json \
  --model ${ASVD_MODEL_7B} \
  --trust_remote_code \
  --tasks blimp \
  --batch_size ${BATCH_SIZE} \
  --device ${DEVICE} \
  --dtype ${LING_DTYPE} \
  --output_json outputs/asvd/jeffwan_llama-7b-hf_asvd40_ling.json

# jeffwan llama-13b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU_BENCH_13B} python -m eval_results.eval_benchmarks \
  --model ${ASVD_MODEL_13B} \
  --use_lm_eval \
  --lm_eval_tasks ${LM_EVAL_TASKS} \
  --lm_eval_num_fewshot 0 \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --batch_size ${BATCH_SIZE} \
  --output_json outputs/asvd/jeffwan_llama-13b-hf_asvd40_lm_eval.json

CUDA_VISIBLE_DEVICES=${GPU_PPL_13B} python -m baselines.ASVD.eval_ASVD_ppl_with_json \
  --checkpoint ${ASVD_MODEL_13B} \
  --datasets ${PPL_DATASETS} \
  --device ${DEVICE} \
  --seqlen 2048 \
  --batch_size ${BATCH_SIZE} \
  --dtype ${DTYPE} \
  --output_json outputs/asvd/jeffwan_llama-13b-hf_asvd40_ppl.json

CUDA_VISIBLE_DEVICES=${GPU_LING_13B} python -m baselines.ASVD.eval_ASVD_linguistic_tasks_with_json \
  --model ${ASVD_MODEL_13B} \
  --trust_remote_code \
  --tasks blimp \
  --batch_size ${BATCH_SIZE} \
  --device ${DEVICE} \
  --dtype ${LING_DTYPE} \
  --output_json outputs/asvd/jeffwan_llama-13b-hf_asvd40_ling.json

# jeffwan llama-30b-hf, 40% params
CUDA_VISIBLE_DEVICES=${GPU_BENCH_30B} python -m eval_results.eval_benchmarks \
  --model ${ASVD_MODEL_30B} \
  --use_lm_eval \
  --lm_eval_tasks ${LM_EVAL_TASKS} \
  --lm_eval_num_fewshot 0 \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --batch_size ${BATCH_SIZE} \
  --output_json outputs/asvd/jeffwan_llama-30b-hf_asvd40_lm_eval.json

CUDA_VISIBLE_DEVICES=${GPU_PPL_30B} python -m baselines.ASVD.eval_ASVD_ppl_with_json \
  --checkpoint ${ASVD_MODEL_30B} \
  --datasets ${PPL_DATASETS} \
  --device ${DEVICE} \
  --seqlen 2048 \
  --batch_size ${BATCH_SIZE} \
  --dtype ${DTYPE} \
  --output_json outputs/asvd/jeffwan_llama-30b-hf_asvd40_ppl.json

CUDA_VISIBLE_DEVICES=${GPU_LING_30B} python -m baselines.ASVD.eval_ASVD_linguistic_tasks_with_json \
  --model ${ASVD_MODEL_30B} \
  --trust_remote_code \
  --tasks blimp \
  --batch_size ${BATCH_SIZE} \
  --device ${DEVICE} \
  --dtype ${LING_DTYPE} \
  --output_json outputs/asvd/jeffwan_llama-30b-hf_asvd40_ling.json
