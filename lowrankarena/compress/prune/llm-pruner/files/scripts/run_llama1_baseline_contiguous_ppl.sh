#!/usr/bin/env bash
set -euo pipefail

OUTPUT_JSON="${1:-/deac/csc/yangGrp/cuij/LLM/llm-pruner/results/ppl_l1_7b/full/llama1_7b_baseline_ppl.json}"
C4_MAX_EVAL_TOKENS="${2:-262144}"

ROOT=/deac/csc/yangGrp/cuij/LLM/llm-pruner
PY=/deac/csc/alqahtaniGrp/cuij/miniconda3/envs/dobisvd/bin/python
BASE_MODEL=/deac/csc/yangGrp/cuij/LLM/models/hf_models/huggyllama__llama-7b

mkdir -p "$(dirname "$OUTPUT_JSON")"
export PYTHONUNBUFFERED=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

CMD=(
  "$PY" "$ROOT/scripts/eval_contiguous_ppl.py"
  --base-model-path "$BASE_MODEL"
  --checkpoint-label "llama1_7b_baseline"
  --output-json "$OUTPUT_JSON"
  --datasets "wikitext2,c4_stream"
  --max-length 2048
  --batch-size 1
  --device cuda:0
  --dtype float16
  --c4-max-eval-tokens "$C4_MAX_EVAL_TOKENS"
)

echo "[run] start $(date)"
echo "[run] target=baseline"
echo "[run] output_json=$OUTPUT_JSON"
echo "[run] c4_max_eval_tokens=$C4_MAX_EVAL_TOKENS"
"${CMD[@]}"
echo "[run] done $(date)"
