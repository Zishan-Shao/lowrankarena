#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="eval_results/template_sensitivity"
mkdir -p "${OUT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"

# Override via env if needed
OUR_CKPT="${OUR_CKPT:-./checkpoints/llama-2-7b-hf_act_lora_lmwhiten_mixedlora_0.4_linguistic_enhanced.pt}"
DOBI_MODEL="${DOBI_MODEL:-Qinsi1/DobiSVD-Llama-2-7b-hf-0.4}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DTYPE="${DTYPE:-bfloat16}"
TASKS="${TASKS:-openbookqa,arc_easy,arc_challenge,winogrande,hellaswag,piqa,mathqa}"
TEMPLATES="${TEMPLATES:-plain,qa,mc_letters,instruction}"
FORCE_RIGHT_PADDING="${FORCE_RIGHT_PADDING:-1}"
FIX_PAD_QUERY_MASK="${FIX_PAD_QUERY_MASK:-1}"
LIMIT="${LIMIT:-}"

extra_flags=()
if [[ "${FORCE_RIGHT_PADDING}" == "1" ]]; then
  extra_flags+=(--force_right_padding)
fi
if [[ "${FIX_PAD_QUERY_MASK}" == "1" ]]; then
  extra_flags+=(--fix_pad_query_mask)
fi

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args+=(--limit "${LIMIT}")
fi

run_one () {
  local name="$1"
  local model="$2"
  local tag
  tag="$(echo "${name}" | tr '/:' '_' | tr ' ' '_' )"
  local out_json="${OUT_DIR}/${tag}_${TS}.json"
  local out_md="${OUT_DIR}/${tag}_${TS}.md"

  echo "[Run] ${name}"
  python eval_template_stability.py \
    --model "${model}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --tasks "${TASKS}" \
    --templates "${TEMPLATES}" \
    --output_json "${out_json}" \
    --output_md "${out_md}" \
    "${extra_flags[@]}" \
    "${limit_args[@]}"
}

run_one "our_checkpoint" "${OUR_CKPT}"
run_one "dobi_checkpoint" "${DOBI_MODEL}"
