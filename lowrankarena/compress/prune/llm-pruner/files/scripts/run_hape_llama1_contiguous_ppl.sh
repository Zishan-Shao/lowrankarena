#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <retain_ratio> <state_dict_checkpoint> [c4_max_eval_tokens]" >&2
  exit 1
fi

TARGET="$1"
CHECKPOINT_PATH="$2"
C4_MAX_EVAL_TOKENS="${3:-262144}"

ROOT=/deac/csc/yangGrp/cuij/LLM/llm-pruner
PY=/deac/csc/alqahtaniGrp/cuij/miniconda3/envs/dobisvd/bin/python
BASE_MODEL=/deac/csc/yangGrp/cuij/LLM/models/hf_models/huggyllama__llama-7b
OUTPUT_JSON="/deac/csc/yangGrp/cuij/LLM/HAP-E/eval_results/llama1_7b/llama1_7b_keep_${TARGET}_ppl.json"
LABEL="llama1_7b_hape_retain_${TARGET}"

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
  --state-dict-checkpoint "$CHECKPOINT_PATH"
)

echo "[run] start $(date)"
echo "[run] target=$TARGET"
echo "[run] checkpoint=$CHECKPOINT_PATH"
echo "[run] output_json=$OUTPUT_JSON"
echo "[run] c4_max_eval_tokens=$C4_MAX_EVAL_TOKENS"
"${CMD[@]}"
echo "[run] done $(date)"
