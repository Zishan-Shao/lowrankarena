#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <gpu_id>" >&2
  exit 1
fi

GPU_ID="$1"
REPO="/deac/csc/yangGrp/cuij/LLM/TransformerCompression"
PYTHON_BIN="/deac/csc/alqahtaniGrp/cuij/miniconda3/envs/dobisvd/bin/python"
MODEL_ID="meta-llama/Llama-2-7b-hf"
RESULT_ROOT="$REPO/results/formal_l1_7b"
SUMMARY_UPDATER="/deac/csc/yangGrp/cuij/LLM/llm-pruner/scripts/update_llama1_section_in_summary.py"
RATIOS=(0.8 0.7 0.6 0.5 0.4)

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$REPO/src:$REPO/.deps:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "[batch] start $(date)"
echo "[batch] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for keep in "${RATIOS[@]}"; do
  sparsity="$($PYTHON_BIN - <<PY
keep = float("$keep")
print(f"{1.0 - keep:.1f}")
PY
)"

  comp_dir="$RESULT_ROOT/keep_${keep}/compressed"
  model_pt="$comp_dir/Llama-2-7b-hf_${keep}.pt"
  eval_dir="$RESULT_ROOT/keep_${keep}/eval_ppl_c4"
  log_file="$eval_dir/c4_ppl.log"
  out_path="$eval_dir/metrics.json"
  mkdir -p "$eval_dir"

  if [[ -f "$out_path" ]]; then
    echo "[skip] keep=$keep metrics already exist $(date)"
    continue
  fi

  until [[ -f "$model_pt" ]]; do
    echo "[wait] keep=$keep waiting for $model_pt $(date)"
    sleep 60
  done

  echo "[start] keep=$keep sparsity=$sparsity $(date)"
  "$PYTHON_BIN" "$REPO/experiments/run_slicegpt.py" \
    --model "$MODEL_ID" \
    --sliced-model-path "$comp_dir" \
    --sparsity "$sparsity" \
    --cal-dataset c4 \
    --ppl-only \
    --ppl-eval-batch-size 2 \
    --device cuda \
    --no-wandb > "$log_file" 2>&1

  "$PYTHON_BIN" - <<PY
import json, pathlib, re, sys
keep = "$keep"
log_path = pathlib.Path(r"$log_file")
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
out_path = pathlib.Path(r"$out_path")
out_path.write_text(json.dumps(metrics, indent=2) + "\\n")
print(json.dumps(metrics, indent=2))
PY

  "$SUMMARY_UPDATER"
  echo "[done] keep=$keep $(date)"
done

echo "[batch] all done $(date)"
