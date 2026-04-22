#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <retain_ratio> <gpu_id>" >&2
  exit 1
fi

RETAIN="$1"
GPU_ID="$2"

ROOT="/deac/csc/yangGrp/cuij/LLM/llm-pruner"
LOG_DIR="$ROOT/logs/local_llama1_eval_bundle"
PPL_DIR="$ROOT/results/ppl_l1_7b/full"
SUMMARY_UPDATER="$ROOT/scripts/update_llama1_section_in_summary.py"

case "$RETAIN" in
  0.8) PRUNE="0.2" ;;
  0.7) PRUNE="0.3" ;;
  0.6) PRUNE="0.4" ;;
  0.5) PRUNE="0.5" ;;
  0.4) PRUNE="0.6" ;;
  *)
    echo "unsupported retain ratio: $RETAIN" >&2
    exit 1
    ;;
esac

MCQ_JSON="$ROOT/results/llama1_7b_r${PRUNE}_pruned_7task.json"
PPL_JSON="$PPL_DIR/llama1_7b_retain_${RETAIN}_ppl.json"

mkdir -p "$LOG_DIR" "$PPL_DIR"

source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
conda activate dobisvd
export PYTHONPATH="$ROOT"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"

cd "$ROOT"

echo "[bundle] start $(date)"
echo "[bundle] retain=$RETAIN prune=$PRUNE gpu=$GPU_ID"

if [[ -f "$MCQ_JSON" ]]; then
  echo "[bundle] mcq skip existing: $MCQ_JSON"
else
  echo "[bundle] mcq start $(date)"
  bash "$ROOT/scripts/run_llama1_pruned_eval_alltasks.sh" "$PRUNE" \
    2>&1 | tee "$LOG_DIR/llama1_retain_${RETAIN}_mcq_g${GPU_ID}.log"
  echo "[bundle] mcq done $(date)"
fi

if [[ -f "$PPL_JSON" ]]; then
  echo "[bundle] ppl skip existing: $PPL_JSON"
else
  echo "[bundle] ppl start $(date)"
  bash "$ROOT/scripts/run_llama1_contiguous_ppl.sh" \
    "$RETAIN" "$PPL_JSON" 262144 \
    2>&1 | tee "$LOG_DIR/llama1_retain_${RETAIN}_ppl_g${GPU_ID}.log"
  echo "[bundle] ppl done $(date)"
fi

python "$SUMMARY_UPDATER"
echo "[bundle] summary refreshed $(date)"
echo "[bundle] done $(date)"
