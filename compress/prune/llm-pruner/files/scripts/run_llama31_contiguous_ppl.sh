#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <baseline|0.8|0.7|0.6|0.5|0.4> <output_json> [c4_max_eval_tokens]" >&2
  exit 1
fi

TARGET="$1"
OUTPUT_JSON="$2"
C4_MAX_EVAL_TOKENS="${3:-262144}"

ROOT=/deac/csc/yangGrp/cuij/LLM/llm-pruner
PY=/deac/csc/alqahtaniGrp/cuij/miniconda3/envs/dobisvd/bin/python
BASE_MODEL=/deac/csc/yangGrp/cuij/LLM/models/hf_models/meta-llama__Llama-3.1-8B

case "$TARGET" in
  baseline)
    LABEL="llama31_8b_baseline"
    CHECKPOINT_PATH=""
    ;;
  0.8|0.7|0.6|0.5|0.4)
    case "$TARGET" in
      0.8) PRUNE_RATIO="0.2" ;;
      0.7) PRUNE_RATIO="0.3" ;;
      0.6) PRUNE_RATIO="0.4" ;;
      0.5) PRUNE_RATIO="0.5" ;;
      0.4) PRUNE_RATIO="0.6" ;;
    esac
    LABEL="llama31_8b_retain_${TARGET}"
    CHECKPOINT_PATH="$ROOT/prune_log/l31_8b_r${PRUNE_RATIO}_prune/pytorch_model.bin"
    ;;
  *)
    echo "unsupported target: $TARGET" >&2
    exit 1
    ;;
esac

mkdir -p "$(dirname "$OUTPUT_JSON")"
export PYTHONUNBUFFERED=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

CMD=(
  "$PY" "$ROOT/scripts/eval_contiguous_ppl.py"
  --base-model-path "$BASE_MODEL"
  --checkpoint-label "$LABEL"
  --output-json "$OUTPUT_JSON"
  --datasets "wikitext2,c4_stream"
  --max-length 2048
  --batch-size 1
  --device cuda:0
  --dtype float16
  --c4-max-eval-tokens "$C4_MAX_EVAL_TOKENS"
)

if [[ -n "$CHECKPOINT_PATH" ]]; then
  CMD+=(--checkpoint-path "$CHECKPOINT_PATH")
fi

echo "[run] start $(date)"
echo "[run] target=$TARGET"
echo "[run] output_json=$OUTPUT_JSON"
echo "[run] c4_max_eval_tokens=$C4_MAX_EVAL_TOKENS"
"${CMD[@]}"
echo "[run] done $(date)"
