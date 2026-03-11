#!/bin/bash
set -euo pipefail

# run from repo root:
# bash baselines/SVD-LLM/eval_SVD_LLMv2.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/baselines/SVD-LLM:${PYTHONPATH:-}"

mkdir -p outputs/svdllmv2

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

SVDLLMV2_MODEL_7B=baselines/SVD-LLM/checkpoints/jeffwan_llama_7b_hf_svdllmv2_keep0.4.pt
SVDLLMV2_MODEL_13B=baselines/SVD-LLM/checkpoints/jeffwan_llama_13b_hf_svdllmv2_keep0.4.pt
SVDLLMV2_MODEL_30B=baselines/SVD-LLM/checkpoints/jeffwan_llama_30b_hf_svdllmv2_keep0.4.pt

run_eval() {
  local gpu_bench="$1"
  local gpu_ppl="$2"
  local gpu_ling="$3"
  local checkpoint="$4"
  local base_model="$5"
  local output_tag="$6"

  export SVDLLM_TOKENIZER_MODEL="${base_model}"

  CUDA_VISIBLE_DEVICES="${gpu_bench}" python baselines/SVD-LLM/eval_SVDLLM_benchmark.py \
    --model "${checkpoint}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --use_lm_eval \
    --lm_eval_tasks "${LM_EVAL_TASKS}" \
    --lm_eval_num_fewshot 0 \
    --lm_eval_max_length "${SEQ_LEN}" \
    --force_right_padding \
    --fix_pad_query_mask \
    --output_json "outputs/svdllmv2/${output_tag}_lm_eval.json"

  CUDA_VISIBLE_DEVICES="${gpu_ppl}" python baselines/SVD-LLM/eval_SVDLLM_ppl.py \
    --checkpoint "${checkpoint}" \
    --datasets "${PPL_DATASETS}" \
    --c4_stream \
    --c4_docs "${C4_DOCS}" \
    --seqlen "${SEQ_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --output_json "outputs/svdllmv2/${output_tag}_ppl.json"

  CUDA_VISIBLE_DEVICES="${gpu_ling}" python baselines/SVD-LLM/eval_SVDLLM_linguistic.py \
    --checkpoint "${checkpoint}" \
    --tasks blimp \
    --num_fewshot 0 \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --dtype "${LING_DTYPE}" \
    --output_json "outputs/svdllmv2/${output_tag}_ling.json"
}

# jeffwan llama-7b-hf, keep ratio ~= 0.4
run_eval \
  "${GPU_BENCH_7B}" "${GPU_PPL_7B}" "${GPU_LING_7B}" \
  "${SVDLLMV2_MODEL_7B}" jeffwan/llama-7b-hf jeffwan_llama-7b-hf_svdllmv240

# jeffwan llama-13b-hf, keep ratio ~= 0.4
run_eval \
  "${GPU_BENCH_13B}" "${GPU_PPL_13B}" "${GPU_LING_13B}" \
  "${SVDLLMV2_MODEL_13B}" jeffwan/llama-13b-hf jeffwan_llama-13b-hf_svdllmv240

# jeffwan llama-30b-hf, keep ratio ~= 0.4
run_eval \
  "${GPU_BENCH_30B}" "${GPU_PPL_30B}" "${GPU_LING_30B}" \
  "${SVDLLMV2_MODEL_30B}" jeffwan/llama-30b-hf jeffwan_llama-30b-hf_svdllmv240
