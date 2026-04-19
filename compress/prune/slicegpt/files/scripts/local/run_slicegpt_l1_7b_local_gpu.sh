#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 GPU_ID KEEP_RATIO [KEEP_RATIO ...]" >&2
  exit 1
fi

GPU_ID="$1"
shift
RATIOS=("$@")

REPO="/deac/csc/yangGrp/cuij/LLM/TransformerCompression"
MODEL_ID="meta-llama/Llama-2-7b-hf"
MODEL_PATH="/deac/csc/yangGrp/cuij/LLM/models/hf_models/huggyllama__llama-7b"
RESULT_ROOT="$REPO/results/formal_l1_7b"
LOG_ROOT="$REPO/logs/local_l1_7b"

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "[worker] gpu=$GPU_ID ratios=${RATIOS[*]} start $(date)"

for ratio in "${RATIOS[@]}"; do
  ratio_tag="${ratio/./}"
  out_root="$RESULT_ROOT/keep_${ratio}"
  log_file="$LOG_ROOT/slicegpt_l1_k${ratio_tag}_g${GPU_ID}.log"

  echo "[worker] gpu=$GPU_ID keep=$ratio start $(date)"
  "$REPO/scripts/slurm/run_slicegpt_ratio.sh" \
    "$MODEL_ID" \
    "$MODEL_PATH" \
    "$ratio" \
    "$out_root" \
    2>&1 | tee "$log_file"
  echo "[worker] gpu=$GPU_ID keep=$ratio done $(date)"
done

echo "[worker] gpu=$GPU_ID all done $(date)"
