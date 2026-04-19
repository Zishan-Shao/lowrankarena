#!/usr/bin/env bash
set -euo pipefail

REPO="/deac/csc/yangGrp/cuij/LLM/TransformerCompression"
PYTHON_BIN="/deac/csc/alqahtaniGrp/cuij/miniconda3/envs/dobisvd/bin/python"
MODEL_ID="meta-llama/Llama-3.1-8B"
RESULT_ROOT="$REPO/results/formal_l31_8b"
RATIOS=(0.8 0.7 0.6 0.5 0.4)

export PYTHONPATH="$REPO/src:$REPO/.deps:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

for keep in "${RATIOS[@]}"; do
  sparsity="$($PYTHON_BIN - <<PY
keep = float('$keep')
print(f"{1.0 - keep:.1f}")
PY
)"

  comp_dir="$RESULT_ROOT/keep_${keep}/compressed"
  eval_dir="$RESULT_ROOT/keep_${keep}/eval_ppl_c4"
  log_file="$eval_dir/c4_ppl.log"
  mkdir -p "$eval_dir"

  echo "[START] keep=$keep sparsity=$sparsity"
  $PYTHON_BIN "$REPO/experiments/run_slicegpt.py" \
    --model "$MODEL_ID" \
    --sliced-model-path "$comp_dir" \
    --sparsity "$sparsity" \
    --cal-dataset c4 \
    --ppl-only \
    --ppl-eval-batch-size 2 \
    --device cuda \
    --no-wandb > "$log_file" 2>&1

  $PYTHON_BIN - <<PY
import json, pathlib, re, sys
keep = "$keep"
log_path = pathlib.Path(r"$log_file")
text = log_path.read_text()
match = re.search(r"Loaded model perplexity: ([0-9.]+)", text)
if not match:
    print(f"Failed to parse C4 PPL from {log_path}", file=sys.stderr)
    sys.exit(1)
metrics = {
    "model": "llama31_8b",
    "method": "slicegpt",
    "retain_ratio": float(keep),
    "dataset": "c4",
    "ppl": float(match.group(1)),
    "log": str(log_path),
}
out_path = pathlib.Path(r"$eval_dir") / "metrics.json"
out_path.write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2))
PY
  echo "[DONE] keep=$keep"
done
