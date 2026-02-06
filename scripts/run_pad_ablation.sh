#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flashsvd

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/eval_results/template_sensitivity"
mkdir -p "${OUT_DIR}"

# Node 2 default: use GPU 2 unless overridden
GPU_ID="${GPU_ID:-2}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DTYPE="${DTYPE:-bfloat16}"
PYTHON="${PYTHON:-python}"
TASKS="${TASKS:-openbookqa,arc_easy,arc_challenge,winogrande,hellaswag,piqa,mathqa}"
TEMPLATES="${TEMPLATES:-plain,qa,mc_letters,instruction}"
PROFILES="${PROFILES:-realistic,rebuttal}"
LIMIT="${LIMIT:-}"

# Full padding sensitivity by default
PAD_ABLATION="${PAD_ABLATION:-full}"

OUR_MODEL="${OUR_MODEL:-${ROOT_DIR}/checkpoints/llama_2_7b_hf_act_lora_mixedwhiten_mixedlora_0.4_enhanced.pt}"
DOBI_MODEL="${DOBI_MODEL:-Qinsi1/DobiSVD-Llama-2-7b-hf-0.4}"

EXTRA_FLAGS=(--force_right_padding --fix_pad_query_mask)

LIMIT_FLAGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_FLAGS=(--limit "${LIMIT}")
fi

TS="$(date +"%Y%m%d_%H%M%S")"

run_eval() {
  local name="$1"
  local model="$2"
  local profile="$3"
  local out_json="${OUT_DIR}/${name}_${TS}.json"
  local out_md="${OUT_DIR}/${name}_${TS}.md"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" "${ROOT_DIR}/eval_template_stability.py" \
    --model "${model}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --tasks "${TASKS}" \
    --templates "${TEMPLATES}" \
    --template_profile "${profile}" \
    --pad_ablation "${PAD_ABLATION}" \
    --output_json "${out_json}" \
    --output_md "${out_md}" \
    "${LIMIT_FLAGS[@]}" \
    "${EXTRA_FLAGS[@]}"
}

IFS=',' read -r -a PROFILE_LIST <<< "${PROFILES}"
for profile in "${PROFILE_LIST[@]}"; do
  profile="${profile//[[:space:]]/}"
  run_eval "ours_pad_ablation_${PAD_ABLATION}_${profile}" "${OUR_MODEL}" "${profile}"
  run_eval "dobi_pad_ablation_${PAD_ABLATION}_${profile}" "${DOBI_MODEL}" "${profile}"
done
