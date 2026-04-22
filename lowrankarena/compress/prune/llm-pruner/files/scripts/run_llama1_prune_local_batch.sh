#!/usr/bin/env bash
set -euo pipefail

ROOT="/deac/csc/yangGrp/cuij/LLM/llm-pruner"
LOG_DIR="$ROOT/logs/local_llama1_prune_batch"
RATIOS=(0.2 0.3 0.4 0.5 0.6)

mkdir -p "$LOG_DIR"

source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
conda activate dobisvd
export PYTHONPATH="$ROOT"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

cd "$ROOT"

echo "[batch] start $(date)"
echo "[batch] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for ratio in "${RATIOS[@]}"; do
  echo "[batch] ratio=${ratio} start $(date)"
  bash "$ROOT/scripts/run_llama1_prune_ratio.sh" "$ratio" \
    2>&1 | tee "$LOG_DIR/llama1_r${ratio}.log"
  echo "[batch] ratio=${ratio} done $(date)"
done

echo "[batch] all done $(date)"
