#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 MODEL_ID MODEL_PATH KEEP_RATIO OUT_ROOT [EXTRA_LM_EVAL_ARGS...]" >&2
  exit 1
fi

MODEL_ID="$1"
MODEL_PATH="$2"
KEEP_RATIO="$3"
OUT_ROOT="$4"
shift 4

REPO="/deac/csc/yangGrp/cuij/LLM/TransformerCompression"
PYTHON_BIN="/deac/csc/alqahtaniGrp/cuij/miniconda3/envs/dobisvd/bin/python"
export PYTHONPATH="$REPO/src:$REPO/.deps:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SPARSITY="$($PYTHON_BIN - <<PY
keep = float('$KEEP_RATIO')
print(f"{1.0 - keep:.1f}")
PY
)"

COMPRESSED_DIR="$OUT_ROOT/compressed"
MCQ_DIR="$OUT_ROOT/eval_mcq"
mkdir -p "$COMPRESSED_DIR" "$MCQ_DIR"

cd "$REPO"

$PYTHON_BIN experiments/run_slicegpt.py \
  --model "$MODEL_ID" \
  --model-path "$MODEL_PATH" \
  --save-dir "$COMPRESSED_DIR" \
  --sparsity "$SPARSITY" \
  --device cuda \
  --no-wandb

$PYTHON_BIN experiments/run_lm_eval.py \
  --model "$MODEL_ID" \
  --sliced-model-path "$COMPRESSED_DIR" \
  --sparsity "$SPARSITY" \
  --save-dir "$MCQ_DIR" \
  --no-wandb \
  "$@"
