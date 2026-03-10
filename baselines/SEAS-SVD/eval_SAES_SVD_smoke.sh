#!/bin/bash

# run from repo root:
# bash baselines/SEAS-SVD/eval_SAES_SVD_smoke.sh

mkdir -p outputs/saes/smoke

SMOKE_MODEL=baselines/SEAS-SVD/robust/jeffwan_llama-7b-hf_saes40_smoke
SMOKE_LM_EVAL_TASKS=arc_easy,piqa
SEQ_LEN=2048
BATCH_SIZE=1
DEVICE=cuda
DTYPE=bfloat16
LING_DTYPE=bfloat16
GPU=3

# smoke lm-eval
CUDA_VISIBLE_DEVICES=${GPU} python -m eval_results.eval_benchmarks \
  --model ${SMOKE_MODEL} \
  --saes_svd \
  --saes_base_model jeffwan/llama-7b-hf \
  --use_lm_eval \
  --lm_eval_tasks ${SMOKE_LM_EVAL_TASKS} \
  --lm_eval_num_fewshot 0 \
  --limit 10 \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --batch_size ${BATCH_SIZE} \
  --output_json outputs/saes/smoke/jeffwan_llama-7b-hf_saes40_smoke_lm_eval.json

# smoke ppl
CUDA_VISIBLE_DEVICES=${GPU} python -m eval_results.eval_general_ppl \
  --saes_model ${SMOKE_MODEL} \
  --saes_base_model jeffwan/llama-7b-hf \
  --datasets wikitext2 \
  --seqlen ${SEQ_LEN} \
  --batch_size ${BATCH_SIZE} \
  --max_batches 2 \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --metrics token \
  --output_json outputs/saes/smoke/jeffwan_llama-7b-hf_saes40_smoke_ppl.json

# smoke linguistic eval
CUDA_VISIBLE_DEVICES=${GPU} python -m eval_results.eval_linguistic_tasks \
  --saes_model ${SMOKE_MODEL} \
  --saes_base_model jeffwan/llama-7b-hf \
  --tasks blimp \
  --num_fewshot 0 \
  --limit 20 \
  --batch_size ${BATCH_SIZE} \
  --device ${DEVICE} \
  --dtype ${LING_DTYPE} \
  --output_json outputs/saes/smoke/jeffwan_llama-7b-hf_saes40_smoke_ling.json
