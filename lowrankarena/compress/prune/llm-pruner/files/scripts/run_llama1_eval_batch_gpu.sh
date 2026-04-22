#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <gpu_id> <retain_ratio1> [retain_ratio2 ...]" >&2
  exit 1
fi

GPU_ID="$1"
shift

ROOT="/deac/csc/yangGrp/cuij/LLM/llm-pruner"

for RETAIN in "$@"; do
  echo "[batch] start retain=${RETAIN} gpu=${GPU_ID} $(date)"
  bash "$ROOT/scripts/run_llama1_eval_bundle.sh" "$RETAIN" "$GPU_ID"
  echo "[batch] done retain=${RETAIN} gpu=${GPU_ID} $(date)"
done
