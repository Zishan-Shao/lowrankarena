#!/bin/bash
#SBATCH --job-name=l31_serve
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/shaoz/lowrankarena/logs/slurm/serving_summary/%x-%j.out
#SBATCH --error=/home/shaoz/lowrankarena/logs/slurm/serving_summary/%x-%j.err

set -uo pipefail

REPO_ROOT="${LRA_REPO_ROOT:-/home/shaoz/lowrankarena}"
INDEX_PATH="${SERVING_INDEX_PATH:-$REPO_ROOT/benchmark/speed/serving_summary_llama31_index.csv}"
SUITE="${SERVING_SUITE:-speed/serve_e2e}"
RUN_LABEL="${SERVING_RUN_LABEL:-serving_summary_llama31_a100}"
OUTPUT_DIR="${SERVING_OUTPUT_DIR:-$REPO_ROOT/results/leaderboard/serving_summary/llama31_8b_a100_0p6_${SLURM_JOB_ID:-manual}}"
PORT="${SERVING_PORT:-8000}"
DTYPE="${SERVING_DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${SERVING_GPU_MEMORY_UTILIZATION:-0.6}"
NUM_PROMPTS="${SERVING_NUM_PROMPTS:-64}"
MAX_CONCURRENCY="${SERVING_MAX_CONCURRENCY:-16}"
REQUEST_RATE="${SERVING_REQUEST_RATE:-1.0}"
READY_TIMEOUT="${SERVING_READY_TIMEOUT:-900}"
MAX_MODEL_LEN="${SERVING_MAX_MODEL_LEN:-}"
PROMPT_LENGTHS="${SERVING_PROMPT_LENGTHS:-${SERVING_PROMPT_LENGTH:-}}"
GENERATION_LENGTHS="${SERVING_GENERATION_LENGTHS:-${SERVING_GENERATION_LENGTH:-}}"
ENFORCE_EAGER="${SERVING_ENFORCE_EAGER:-0}"

CHECKPOINTS="${SERVING_CHECKPOINTS:-llama31-8b-dense-fp llama31-8b-asvd-0.6-serving llama31-8b-svdllm-v1-update-0.6-serving llama31-8b-svdllm-v2-0.6-serving llama31-8b-dobi-0.6-serving llama31-8b-basis-sharing-0.6-serving llama31-8b-modegpt-0.6-serving}"

LOG_DIR="$OUTPUT_DIR/logs"
STATUS_FILE="$OUTPUT_DIR/status.tsv"
SUMMARY_TSV="$OUTPUT_DIR/serving_summary.tsv"
SUMMARY_TEX="$OUTPUT_DIR/serving_summary_table.tex"

mkdir -p "$REPO_ROOT/logs/slurm/serving_summary" "$LOG_DIR"

source /home/shaoz/miniconda3/etc/profile.d/conda.sh
conda activate "${LOWRANKARENA_CONDA_ENV:-lowrankarena}"

cd "$REPO_ROOT"

export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo -e "checkpoint\tstatus" > "$STATUS_FILE"
echo "[env] repo=$REPO_ROOT"
echo "[env] index=$INDEX_PATH"
echo "[env] suite=$SUITE output_dir=$OUTPUT_DIR"
echo "[env] dtype=$DTYPE gpu_memory_utilization=$GPU_MEMORY_UTILIZATION request_rate=$REQUEST_RATE num_prompts=$NUM_PROMPTS max_concurrency=$MAX_CONCURRENCY"
nvidia-smi || true
python - <<'PY' || true
import importlib.metadata as metadata
for package in ("torch", "transformers", "vllm", "huggingface_hub"):
    try:
        print(f"[pkg] {package}={metadata.version(package)}", flush=True)
    except metadata.PackageNotFoundError:
        print(f"[pkg] {package}=missing", flush=True)
PY

for CHECKPOINT in $CHECKPOINTS; do
  echo "[run:start] $CHECKPOINT"
  LOG_FILE="$LOG_DIR/${CHECKPOINT}.log"
  EXTRA_ARGS=()
  for PROMPT_LENGTH in $PROMPT_LENGTHS; do
    EXTRA_ARGS+=(--prompt-length "$PROMPT_LENGTH")
  done
  for GENERATION_LENGTH in $GENERATION_LENGTHS; do
    EXTRA_ARGS+=(--generation-length "$GENERATION_LENGTH")
  done
  if [ -n "$MAX_MODEL_LEN" ]; then
    EXTRA_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
  fi
  case "${ENFORCE_EAGER,,}" in
    1|true|yes|y|on)
      EXTRA_ARGS+=(--enforce-eager)
      ;;
  esac
  python "$REPO_ROOT/scripts/run_speed.py" "$CHECKPOINT" \
    --suite "$SUITE" \
    --index "$INDEX_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --request-rate "$REQUEST_RATE" \
    --ready-check-timeout-seconds "$READY_TIMEOUT" \
    --run-label "$RUN_LABEL" \
    "${EXTRA_ARGS[@]}" \
    --strict-validation 2>&1 | tee "$LOG_FILE"
  STATUS="${PIPESTATUS[0]}"
  echo -e "$CHECKPOINT\t$STATUS" >> "$STATUS_FILE"
  echo "[run:done] $CHECKPOINT status=$STATUS"
  sleep 5
done

python - "$OUTPUT_DIR" "$SUMMARY_TSV" "$SUMMARY_TEX" $CHECKPOINTS <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
summary_tsv = Path(sys.argv[2])
summary_tex = Path(sys.argv[3])
checkpoints = sys.argv[4:]

label_by_checkpoint = {
    "llama31-8b-dense-fp": "Dense FP",
    "llama31-8b-asvd-0.6-serving": "ASVD",
    "llama31-8b-svdllm-v1-update-0.6-serving": "\\textsc{SVD-LLM} v1",
    "llama31-8b-svdllm-v2-0.6-serving": "\\textsc{SVD-LLM} v2",
    "llama31-8b-dobi-0.6-serving": "DoBi-SVD",
    "llama31-8b-basis-sharing-0.6-serving": "Basis Sharing",
    "llama31-8b-modegpt-0.6-serving": "MoDeGPT",
}
profiles = [
    "prefill_heavy_4k_to_32",
    "balanced_2k_to_128",
    "decode_heavy_512_to_512",
]

def fmt(value: object, digits: int = 1) -> str:
    if value is None:
        return "--"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "--"
    if numeric != numeric:
        return "--"
    return f"{numeric:.{digits}f}"

def profile_map(payload: dict) -> dict[str, dict]:
    items = payload.get("details", {}).get("profiles", [])
    return {str(item.get("name")): item for item in items if isinstance(item, dict)}

rows = []
for checkpoint in checkpoints:
    method = label_by_checkpoint.get(checkpoint, checkpoint)
    result_path = output_dir / f"speed__serve_e2e__{checkpoint}.json"
    legacy_result_path = output_dir / f"serve_e2e__{checkpoint}.json"
    if not result_path.exists() and legacy_result_path.exists():
        result_path = legacy_result_path
    profile_payloads = {}
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        profile_payloads = profile_map(payload)
    row = {"method": method, "checkpoint": checkpoint, "result_path": str(result_path) if result_path.exists() else ""}
    for profile in profiles:
        item = profile_payloads.get(profile, {})
        wall_time = float(item.get("wall_time_seconds") or 0.0)
        total_input = float(item.get("total_input_tokens") or 0.0)
        ttft = item.get("ttft_ms", {}) if isinstance(item.get("ttft_ms"), dict) else {}
        e2e = item.get("e2e_latency_ms", {}) if isinstance(item.get("e2e_latency_ms"), dict) else {}
        row[f"{profile}:ttft_ms_mean"] = ttft.get("mean")
        row[f"{profile}:e2e_ms_mean"] = e2e.get("mean")
        row[f"{profile}:input_tokens_per_second"] = total_input / wall_time if wall_time > 0 else None
        row[f"{profile}:output_tokens_per_second"] = item.get("output_tokens_per_second")
    rows.append(row)

fieldnames = ["method", "checkpoint", "result_path"]
for profile in profiles:
    fieldnames += [
        f"{profile}:ttft_ms_mean",
        f"{profile}:e2e_ms_mean",
        f"{profile}:input_tokens_per_second",
        f"{profile}:output_tokens_per_second",
    ]
with summary_tsv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

lines = []
for row in rows:
    cells = [row["method"]]
    for profile in profiles:
        cells.extend(
            [
                fmt(row[f"{profile}:ttft_ms_mean"]),
                fmt(row[f"{profile}:e2e_ms_mean"]),
                fmt(row[f"{profile}:input_tokens_per_second"], digits=0),
                fmt(row[f"{profile}:output_tokens_per_second"], digits=0),
            ]
        )
    lines.append("        " + " & ".join(cells) + r" \\")

summary_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[summary] wrote {summary_tsv}", flush=True)
print(f"[summary] wrote {summary_tex}", flush=True)
PY

echo "[done] serving summary run finished"
