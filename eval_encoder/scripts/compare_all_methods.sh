#!/usr/bin/env bash
################################################################################
# compare_all_methods.sh
#
# Run multiple compression methods across ALL GLUE tasks and summarize results.
#
# Default: Test all 5 methods (dense, svd, fwsvd, drone, adasvd) on all 8 GLUE
# tasks without fine-tuning (quick test to verify pipeline works).
#
# Optional: Set TWO_STAGE=true to compare "no_finetune" vs "with_finetune"
#
# Usage examples:
#   # Quick test (no finetune, naive backend only)
#   bash eval_encoder/scripts/compare_all_methods.sh
#
#   # Full test (both stages, both backends)
#   TWO_STAGE=true BACKENDS="flashsvd naive" bash eval_encoder/scripts/compare_all_methods.sh
#
#   # Test specific tasks only
#   TASKS="rte mrpc cola" bash eval_encoder/scripts/compare_all_methods.sh
#
# Results: Paper-style table (8 GLUE tasks + G-Avg/A-Avg) from JSON artifacts
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

# ------------------------- user-tunable defaults ------------------------------
# Methods: dense, svd, fwsvd, drone, adasvd
RANK="${RANK:-256}"
RANK_ATTN="${RANK_ATTN:-}"        # Component-specific ranks (optional)
RANK_FFN="${RANK_FFN:-}"
RANK_WO="${RANK_WO:-}"
QKV_MODE="${QKV_MODE:-per_head}"  # per_head or full
CALIB_BATCHES="${CALIB_BATCHES:-16}"  # Calibration batches for fwsvd/drone/adasvd (increased from 4)
BUDGET="${BUDGET:-0.5}"

# Tasks (space-separated, required by glue_pipeline.py nargs="+")
# All 8 GLUE tasks by default
TASKS="${TASKS:-cola sst2 mrpc qqp mnli qnli rte stsb}"

# Whether to start from task-specific fine-tuned checkpoints (recommended)
USE_TASK_MODELS="${USE_TASK_MODELS:-true}"
TASK_MODEL_PREFIX="${TASK_MODEL_PREFIX:-textattack}"

# Output locations
RESULT_DIR="${RESULT_DIR:-eval_encoder/glue_results}"
LOG_DIR="${LOG_DIR:-eval_encoder/eval_results}"
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

# Two-stage by default: compare "no finetune" vs "with finetune"
# Set to false for quick testing (no finetune only)
TWO_STAGE="${TWO_STAGE:-false}"

# Backends to test
# For quick test: only naive (faster, works for all methods)
# For full test: "flashsvd naive" (test both backends)
BACKENDS="${BACKENDS:-naive}"
MODEL_ID="${MODEL_ID:-bert-base-uncased}"

# ------------------------- derived settings -----------------------------------
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${LOG_DIR}/compare_all_${TIMESTAMP}.log"

echo "════════════════════════════════════════════════════════════════════" | tee "${SUMMARY_LOG}"
echo "Compare All Methods (GLUE)  |  ${TIMESTAMP}" | tee -a "${SUMMARY_LOG}"
echo "Repo: ${REPO_ROOT}" | tee -a "${SUMMARY_LOG}"
echo "Tasks: ${TASKS}" | tee -a "${SUMMARY_LOG}"
echo "USE_TASK_MODELS=${USE_TASK_MODELS}  TASK_MODEL_PREFIX=${TASK_MODEL_PREFIX}" | tee -a "${SUMMARY_LOG}"
echo "RANK=${RANK}  RANK_ATTN=${RANK_ATTN}  RANK_FFN=${RANK_FFN}  RANK_WO=${RANK_WO}" | tee -a "${SUMMARY_LOG}"
echo "QKV_MODE=${QKV_MODE}  CALIB_BATCHES=${CALIB_BATCHES}" | tee -a "${SUMMARY_LOG}"
echo "BUDGET=${BUDGET}  BACKENDS=${BACKENDS}  MODEL_ID=${MODEL_ID}" | tee -a "${SUMMARY_LOG}"
echo "TWO_STAGE=${TWO_STAGE}" | tee -a "${SUMMARY_LOG}"
echo "════════════════════════════════════════════════════════════════════" | tee -a "${SUMMARY_LOG}"
echo "" | tee -a "${SUMMARY_LOG}"

# Methods configuration: method_name + extra env vars
METHODS=(
  "dense"
  "svd"
  "fwsvd"
  "drone"
  "adasvd"
)

# Stages
# Default: only test without fine-tuning (quick test)
STAGES=("no_finetune")
if [[ "${TWO_STAGE}" == "true" ]]; then
  # Full test: both stages
  STAGES=("no_finetune" "with_finetune")
fi

# We'll collect JSON paths per (stage, backend, method)
declare -A JSON_BY_STAGE_BACKEND_METHOD

# Helper: find latest JSON for method+backend from RESULT_DIR
latest_json_for_method() {
  local method="$1"
  local backend="$2"
  ls -t "${RESULT_DIR}/glue_results_${method}_${backend}_"*.json 2>/dev/null | head -n 1 || true
}

run_one() {
  local stage="$1"
  local backend="$2"
  local method="$3"

  echo "────────────────────────────────────────────────────────────────────" | tee -a "${SUMMARY_LOG}"
  echo "Stage=${stage}  Backend=${backend}  Method=${method}" | tee -a "${SUMMARY_LOG}"
  echo "────────────────────────────────────────────────────────────────────" | tee -a "${SUMMARY_LOG}"

  local skip_finetuning="false"
  local reuse_checkpoint="false"

  if [[ "${stage}" == "no_finetune" ]]; then
    skip_finetuning="true"
    reuse_checkpoint="false"  # 阶段1：重新压缩
  elif [[ "${stage}" == "with_finetune" ]]; then
    skip_finetuning="false"
    reuse_checkpoint="true"   # 阶段2：重用阶段1的checkpoint，避免重复压缩
  fi

  # Method-specific env
  local env_prefix=()
  env_prefix+=("METHOD=${method}")
  env_prefix+=("TASKS=${TASKS}")
  env_prefix+=("MODEL_ID=${MODEL_ID}")
  env_prefix+=("BACKEND=${backend}")
  env_prefix+=("USE_TASK_MODELS=${USE_TASK_MODELS}")
  env_prefix+=("TASK_MODEL_PREFIX=${TASK_MODEL_PREFIX}")
  env_prefix+=("SKIP_FINETUNING=${skip_finetuning}")
  env_prefix+=("SKIP_COMPRESSION=false")
  env_prefix+=("REUSE_CHECKPOINT=${reuse_checkpoint}")
  env_prefix+=("NON_INTERACTIVE=true")

  # Add component-specific ranks if set
  if [[ -n "${RANK_ATTN}" ]]; then
    env_prefix+=("RANK_ATTN=${RANK_ATTN}")
  fi
  if [[ -n "${RANK_FFN}" ]]; then
    env_prefix+=("RANK_FFN=${RANK_FFN}")
  fi
  if [[ -n "${RANK_WO}" ]]; then
    env_prefix+=("RANK_WO=${RANK_WO}")
  fi

  # Add QKV mode and calibration batches
  env_prefix+=("QKV_MODE=${QKV_MODE}")
  env_prefix+=("CALIB_BATCHES=${CALIB_BATCHES}")

  if [[ "${method}" == "adasvd" ]]; then
    env_prefix+=("BUDGET=${BUDGET}")
  elif [[ "${method}" == "svd" || "${method}" == "fwsvd" || "${method}" == "drone" ]]; then
    env_prefix+=("RANK=${RANK}")
  fi

  # Run
  local start_ts end_ts dur
  start_ts=$(date +%s)
  (
    export "${env_prefix[@]}"
    bash eval_encoder/scripts/one_click_glue.sh
  ) 2>&1 | tee -a "${SUMMARY_LOG}"
  end_ts=$(date +%s)
  dur=$((end_ts - start_ts))
  echo "✅ Done: stage=${stage} backend=${backend} method=${method} (${dur}s)" | tee -a "${SUMMARY_LOG}"

  # Collect JSON
  local js
  js="$(latest_json_for_method "${method}" "${backend}")"
  if [[ -n "${js}" ]]; then
    JSON_BY_STAGE_BACKEND_METHOD["${stage}:${backend}:${method}"]="${js}"
    echo "[collect] ${stage}:${backend}:${method} => ${js}" | tee -a "${SUMMARY_LOG}"
  else
    echo "[collect][warn] No JSON found for ${stage}:${backend}:${method} under ${RESULT_DIR}" | tee -a "${SUMMARY_LOG}"
  fi

  echo "" | tee -a "${SUMMARY_LOG}"
}

# ------------------------- main loops -----------------------------------------
SUCCESS=0
FAIL=0

for stage in "${STAGES[@]}"; do
  for backend in ${BACKENDS}; do
    for method in "${METHODS[@]}"; do
      if run_one "${stage}" "${backend}" "${method}"; then
        SUCCESS=$((SUCCESS + 1))
      else
        FAIL=$((FAIL + 1))
      fi
    done
  done
done

# ------------------------- summary tables -------------------------------------
# Export associative array as JSON string for python
JSON_BY_STAGE_BACKEND_METHOD_JSON="{"
first=1
for k in "${!JSON_BY_STAGE_BACKEND_METHOD[@]}"; do
  v="${JSON_BY_STAGE_BACKEND_METHOD[$k]}"
  if [[ $first -eq 0 ]]; then JSON_BY_STAGE_BACKEND_METHOD_JSON+=","
  else first=0
  fi
  # basic escaping for quotes/backslashes
  esc_k="${k//\\/\\\\}"; esc_k="${esc_k//\"/\\\"}"
  esc_v="${v//\\/\\\\}"; esc_v="${esc_v//\"/\\\"}"
  JSON_BY_STAGE_BACKEND_METHOD_JSON+="\"${esc_k}\":\"${esc_v}\""
done
JSON_BY_STAGE_BACKEND_METHOD_JSON+="}"
export JSON_BY_STAGE_BACKEND_METHOD_JSON

echo "════════════════════════════════════════════════════════════════════" | tee -a "${SUMMARY_LOG}"
echo "Paper-style GLUE summary (from JSON artifacts)" | tee -a "${SUMMARY_LOG}"
echo "════════════════════════════════════════════════════════════════════" | tee -a "${SUMMARY_LOG}"
echo "" | tee -a "${SUMMARY_LOG}"

python3 - <<'PY' 2>&1 | tee -a "${SUMMARY_LOG}"
import os, json

raw = os.environ.get("JSON_BY_STAGE_BACKEND_METHOD_JSON", "{}")
try:
    mp = json.loads(raw)
except Exception as e:
    print("[error] cannot parse JSON_BY_STAGE_BACKEND_METHOD_JSON:", e)
    print(raw)
    raise SystemExit(1)

tasks_order = ["cola","sst2","mrpc","qqp","mnli","qnli","rte","stsb"]
task_hdr = {
    "cola": "CoLA(MCC)",
    "sst2": "SST-2(Acc)",
    "mrpc": "MRPC(F1)",
    "qqp":  "QQP(F1)",
    "mnli": "MNLI(Acc)",
    "qnli": "QNLI(Acc)",
    "rte":  "RTE(Acc)",
    "stsb": "STS-B(Pearson)",
}

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def best_value_for_task(res):
    # glue_pipeline emits best_value
    if not res:
        return None
    v = res.get("best_value")
    if v is None:
        # fallback: use final of primary_metric
        metrics = res.get("metrics", {})
        pm = metrics.get("primary_metric")
        if pm and "final" in metrics and pm in metrics["final"]:
            return float(metrics["final"][pm])
        return None
    return float(v)

def stage_backend_maps(mp):
    """Parse stage:backend:method keys into nested dict"""
    out = {}
    for k, v in mp.items():
        parts = k.split(":")
        if len(parts) == 3:
            stage, backend, method = parts
            out.setdefault(stage, {}).setdefault(backend, {})[method] = v
    return out

def print_table(stage, backend, mm):
    methods_order = ["dense","svd","fwsvd","drone","adasvd"]
    headers = ["Method"] + [task_hdr[t] for t in tasks_order] + ["G-Avg","A-Avg","JSON"]
    print("="*110)
    print(f"Stage: {stage}  |  Backend: {backend}")
    print("="*110)
    print(" | ".join(headers))
    print("-"*110)
    for m in methods_order:
        path = mm.get(m)
        if not path or not os.path.exists(path):
            print(m + " | " + " | ".join(["(missing)"]*(len(headers)-1)))
            continue
        obj = load(path)
        results = {r.get("task"): r for r in obj.get("results", [])}
        vals = []
        for t in tasks_order:
            vals.append(best_value_for_task(results.get(t)))
        summ = obj.get("summary", {})
        gavg = float(summ.get("G-Avg", {}).get("final", 0.0))
        aavg = float(summ.get("A-Avg", {}).get("final", 0.0))
        line = [m] + [("-" if v is None else f"{v:.4f}") for v in vals] + [f"{gavg:.4f}", f"{aavg:.4f}", os.path.basename(path)]
        print(" | ".join(line))
    print()

nested = stage_backend_maps(mp)
for stage in sorted(nested.keys()):
    for backend in sorted(nested[stage].keys()):
        print_table(stage, backend, nested[stage][backend])
PY

echo "Log saved to: ${SUMMARY_LOG}"
echo "Success: ${SUCCESS}  Fail: ${FAIL}"
if [[ "${FAIL}" -gt 0 ]]; then
  exit 1
fi
exit 0
