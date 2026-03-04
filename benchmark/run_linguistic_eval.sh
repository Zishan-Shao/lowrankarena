#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flashsvd

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/eval_results/linguistic"
mkdir -p "${OUT_DIR}"

GPU_ID="${GPU_ID:-3}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DTYPE="${DTYPE:-bfloat16}"
TASKS="${TASKS:-blimp,cola}"
EXTRA_TASKS="${EXTRA_TASKS:-mela_en,lingoly,zhoblimp}"
TASK_SETS="${TASK_SETS:-base:blimp,cola;full:blimp,cola,mela_en,lingoly,zhoblimp}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
LIMIT="${LIMIT:-}"

OUR_MODEL="${OUR_MODEL:-${ROOT_DIR}/checkpoints/mixed_calibrate/llama_2_7b_hf_act_lora_mixedwhiten_mixedlora_0.4_enhanced.pt}"
DOBI_MODEL="${DOBI_MODEL:-Qinsi1/DobiSVD-Llama-2-7b-hf-0.4}"

LIMIT_FLAGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_FLAGS=(--limit "${LIMIT}")
fi

TS="$(date +"%Y%m%d_%H%M%S")"

run_eval() {
  local name="$1"
  local mode="$2"
  local model="$3"
  local out_json="${OUT_DIR}/${name}_${TS}.json"
  local out_md="${OUT_DIR}/${name}_${TS}.md"

  local task_flags=()
  if [[ -n "${TASK_SETS}" ]]; then
    task_flags=(--task_sets "${TASK_SETS}")
  else
    task_flags=(--tasks "${TASKS}" --extra_tasks "${EXTRA_TASKS}")
  fi

  CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${ROOT_DIR}/eval_linguistic_tasks.py" \
    "${mode}" "${model}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    "${task_flags[@]}" \
    --num_fewshot "${NUM_FEWSHOT}" \
    --output_json "${out_json}" \
    --output_md "${out_md}" \
    "${LIMIT_FLAGS[@]}"
}

run_eval "ours_linguistic" "--checkpoint" "${OUR_MODEL}"
run_eval "dobi_linguistic" "--dobi_model" "${DOBI_MODEL}"
