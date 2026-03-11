#!/bin/bash
set -euo pipefail

# run from repo root:
# bash baselines/SVD-LLM/eval_SVD_LLM_smoke.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"


export SVDLLM_TOKENIZER_MODEL="${SVDLLM_TOKENIZER_MODEL:-jeffwan/llama-7b-hf}"

mkdir -p outputs/svdllm/smoke

SMOKE_MODEL=baselines/SVD-LLM/checkpoints_smoke/jeffwan_llama_7b_hf_whitening_only_0.4.pt
SMOKE_LM_EVAL_TASKS=arc_easy,piqa
SEQ_LEN=2048
BATCH_SIZE=1
DEVICE=cuda
DTYPE=bfloat16
LING_DTYPE=bfloat16
GPU=6

# smoke lm-eval
CUDA_VISIBLE_DEVICES="${GPU}" python eval_results/eval_benchmarks.py \
  --model "${SMOKE_MODEL}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --dtype "${DTYPE}" \
  --use_lm_eval \
  --lm_eval_tasks "${SMOKE_LM_EVAL_TASKS}" \
  --lm_eval_num_fewshot 0 \
  --lm_eval_max_length "${SEQ_LEN}" \
  --limit 10 \
  --force_right_padding \
  --fix_pad_query_mask \
  --output_json outputs/svdllm/smoke/jeffwan_llama-7b-hf_svdllm40_smoke_lm_eval.json

# smoke ppl
CUDA_VISIBLE_DEVICES="${GPU}" python eval_results/eval_general_ppl.py \
  --checkpoint "${SMOKE_MODEL}" \
  --datasets wikitext2 \
  --max_batches 2 \
  --seqlen "${SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --output_json outputs/svdllm/smoke/jeffwan_llama-7b-hf_svdllm40_smoke_ppl.json

# smoke linguistic eval
CUDA_VISIBLE_DEVICES="${GPU}" python eval_results/eval_linguistic_tasks.py \
  --checkpoint "${SMOKE_MODEL}" \
  --tasks blimp \
  --limit 20 \
  --num_fewshot 0 \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${LING_DTYPE}" \
  --output_json outputs/svdllm/smoke/jeffwan_llama-7b-hf_svdllm40_smoke_ling.json
