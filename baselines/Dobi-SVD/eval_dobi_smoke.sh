#!/bin/bash

# run from repo root:
# bash baselines/Dobi-SVD/eval_dobi_smoke.sh

mkdir -p outputs/dobi/smoke

SMOKE_MODEL=baselines/Dobi-SVD/results_smoke/compressed_model/llama-7b-hf/DobiSVD-llama-7b-hf-0.4
SMOKE_LM_EVAL_TASKS=arc_easy,piqa
DEVICE=cuda
DTYPE=bfloat16
LING_DTYPE=bfloat16
BATCH_SIZE=1
GPU=6

# smoke lm-eval
CUDA_VISIBLE_DEVICES=${GPU} python -m eval_results.eval_benchmarks \
  --dobi_model ${SMOKE_MODEL} \
  --dobi_remapping \
  --device ${DEVICE} \
  --batch_size ${BATCH_SIZE} \
  --use_lm_eval \
  --dtype ${DTYPE} \
  --lm_eval_tasks ${SMOKE_LM_EVAL_TASKS} \
  --limit 10 \
  --output_json outputs/dobi/smoke/jeffwan_llama-7b-hf_dobi40_smoke_lm_eval.json

# smoke ppl
CUDA_VISIBLE_DEVICES=${GPU} python -m eval_results.eval_general_ppl \
  --dobi_model ${SMOKE_MODEL} \
  --dobi_remapping \
  --datasets wikitext2 \
  --metrics token \
  --max_batches 2 \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --output_json outputs/dobi/smoke/jeffwan_llama-7b-hf_dobi40_smoke_ppl.json

# smoke linguistic eval
CUDA_VISIBLE_DEVICES=${GPU} python -m eval_results.eval_linguistic_tasks \
  --dobi_model ${SMOKE_MODEL} \
  --dobi_remapping \
  --tasks blimp \
  --limit 20 \
  --device ${DEVICE} \
  --dtype ${LING_DTYPE} \
  --output_json outputs/dobi/smoke/jeffwan_llama-7b-hf_dobi40_smoke_ling.json
