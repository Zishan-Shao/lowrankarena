#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <gpu_id> <keep_ratio>" >&2
  exit 1
fi

GPU_ID="$1"
KEEP="$2"

REPO="/deac/csc/yangGrp/cuij/LLM/TransformerCompression"
PYTHON_BIN="/deac/csc/alqahtaniGrp/cuij/miniconda3/envs/dobisvd/bin/python"
MODEL_ID="meta-llama/Llama-2-7b-hf"
RESULT_ROOT="$REPO/results/formal_l1_7b"
SUMMARY_UPDATER="/deac/csc/yangGrp/cuij/LLM/llm-pruner/scripts/update_llama1_section_in_summary.py"

SPARSITY="$($PYTHON_BIN - <<PY
keep = float("$KEEP")
print(f"{1.0 - keep:.1f}")
PY
)"

COMP_DIR="$RESULT_ROOT/keep_${KEEP}/compressed"
MODEL_PT="$COMP_DIR/Llama-2-7b-hf_${KEEP}.pt"
EVAL_DIR="$RESULT_ROOT/keep_${KEEP}/eval_ppl_c4"
LOG_FILE="$EVAL_DIR/c4_ppl.log"
OUT_PATH="$EVAL_DIR/metrics.json"

mkdir -p "$EVAL_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$REPO/src:$REPO/.deps:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "[run] start $(date)"
echo "[run] gpu=$GPU_ID keep=$KEEP sparsity=$SPARSITY"
echo "[run] model_pt=$MODEL_PT"

if [[ -f "$OUT_PATH" ]]; then
  echo "[run] skip existing metrics: $OUT_PATH"
  exit 0
fi

if [[ ! -f "$MODEL_PT" ]]; then
  echo "[run] missing model checkpoint: $MODEL_PT" >&2
  exit 1
fi

"$PYTHON_BIN" "$REPO/experiments/run_slicegpt.py" \
  --model "$MODEL_ID" \
  --sliced-model-path "$COMP_DIR" \
  --sparsity "$SPARSITY" \
  --cal-dataset c4 \
  --ppl-only \
  --ppl-eval-batch-size 2 \
  --device cuda \
  --no-wandb > "$LOG_FILE" 2>&1

"$PYTHON_BIN" - <<PY
import json, pathlib, re, sys
keep = "$KEEP"
log_path = pathlib.Path(r"$LOG_FILE")
text = log_path.read_text()
match = re.search(r"Loaded model perplexity: ([0-9.]+)", text)
if not match:
    print(f"Failed to parse C4 PPL from {log_path}", file=sys.stderr)
    sys.exit(1)
metrics = {
    "model": "llama1_7b",
    "method": "slicegpt",
    "retain_ratio": float(keep),
    "dataset": "c4",
    "ppl": float(match.group(1)),
    "log": str(log_path),
}
out_path = pathlib.Path(r"$OUT_PATH")
out_path.write_text(json.dumps(metrics, indent=2) + "\\n")
print(json.dumps(metrics, indent=2))
PY

"$PYTHON_BIN" "$SUMMARY_UPDATER"
echo "[run] done $(date)"
