#!/bin/bash

# run from repo root:
# bash baselines/ASVD/ASVD_eval_smoke.sh

mkdir -p outputs/asvd/smoke

SMOKE_MODEL=baselines/ASVD/huggingface_repos/jeffwan_llama-7b-hf_asvd40_smoke
SMOKE_LM_EVAL_TASKS=arc_easy,piqa
DEVICE=cuda
DTYPE=bfloat16
LING_DTYPE=fp16
BATCH_SIZE=1
GPU=5

# smoke lm-eval
CUDA_VISIBLE_DEVICES=${GPU} python -m eval_results.eval_benchmarks \
  --model ${SMOKE_MODEL} \
  --use_lm_eval \
  --lm_eval_tasks ${SMOKE_LM_EVAL_TASKS} \
  --lm_eval_num_fewshot 0 \
  --limit 10 \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --batch_size ${BATCH_SIZE} \
  --output_json outputs/asvd/smoke/jeffwan_llama-7b-hf_asvd40_smoke_lm_eval.json

# smoke ppl
CUDA_VISIBLE_DEVICES=${GPU} python -m baselines.ASVD.eval_ASVD_ppl_with_json \
  --checkpoint ${SMOKE_MODEL} \
  --datasets wikitext2 \
  --max_batches 2 \
  --device ${DEVICE} \
  --seqlen 2048 \
  --batch_size ${BATCH_SIZE} \
  --dtype ${DTYPE} \
  --output_json outputs/asvd/smoke/jeffwan_llama-7b-hf_asvd40_smoke_ppl.json

# smoke linguistic eval
CUDA_VISIBLE_DEVICES=${GPU} python -m baselines.ASVD.eval_ASVD_linguistic_tasks_with_json \
  --model ${SMOKE_MODEL} \
  --trust_remote_code \
  --tasks blimp \
  --limit 20 \
  --batch_size ${BATCH_SIZE} \
  --device ${DEVICE} \
  --dtype ${LING_DTYPE} \
  --output_json outputs/asvd/smoke/jeffwan_llama-7b-hf_asvd40_smoke_ling.json
