#!/usr/bin/env bash
set -euo pipefail

ROOT="/deac/csc/yangGrp/cuij/LLM/llm-pruner"
LOG_DIR="$ROOT/logs/local_llama1_eval_batch"
PPL_DIR="$ROOT/results/ppl_l1_7b/full"
RETAINS=(0.8 0.7 0.6 0.5 0.4)
SUMMARY_UPDATER="$ROOT/scripts/update_llama1_section_in_summary.py"

mkdir -p "$LOG_DIR" "$PPL_DIR"

source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
conda activate dobisvd
export PYTHONPATH="$ROOT"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

cd "$ROOT"

echo "[batch] start $(date)"
echo "[batch] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for retain in "${RETAINS[@]}"; do
  case "$retain" in
    0.8) prune="0.2" ;;
    0.7) prune="0.3" ;;
    0.6) prune="0.4" ;;
    0.5) prune="0.5" ;;
    0.4) prune="0.6" ;;
  esac

  MCQ_JSON="$ROOT/results/llama1_7b_r${prune}_pruned_7task.json"
  PPL_JSON="$PPL_DIR/llama1_7b_retain_${retain}_ppl.json"

  if [[ -f "$MCQ_JSON" ]]; then
    echo "[batch] retain=${retain} prune=${prune} mcq skip existing $(date)"
  else
    echo "[batch] retain=${retain} prune=${prune} mcq start $(date)"
    bash "$ROOT/scripts/run_llama1_pruned_eval_alltasks.sh" "$prune" \
      2>&1 | tee "$LOG_DIR/llama1_retain_${retain}_mcq.log"
    echo "[batch] retain=${retain} prune=${prune} mcq done $(date)"
  fi

  if [[ -f "$PPL_JSON" ]]; then
    echo "[batch] retain=${retain} ppl skip existing $(date)"
  else
    echo "[batch] retain=${retain} ppl start $(date)"
    bash "$ROOT/scripts/run_llama1_contiguous_ppl.sh" \
      "$retain" "$PPL_JSON" \
      262144 2>&1 | tee "$LOG_DIR/llama1_retain_${retain}_ppl.log"
    echo "[batch] retain=${retain} ppl done $(date)"
  fi

  "$SUMMARY_UPDATER"
  echo "[batch] summary refreshed $(date)"
done

echo "[batch] all done $(date)"
