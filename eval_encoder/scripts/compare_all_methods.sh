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
SEQ_LEN="${SEQ_LEN:-128}"             # Input sequence length (128 for standard GLUE, 512 for full)
CALIB_BATCHES="${CALIB_BATCHES:-16}"  # Calibration batches for fwsvd/drone (NOT used by adasvd_origin)
BUDGET="${BUDGET:-0.6}"
ADASVD_CALIB_SAMPLES="${ADASVD_CALIB_SAMPLES:-4000}"  # ARS calibration samples (paper: ~4000)
ADASVD_STEPS="${ADASVD_STEPS:-800}"                   # ARS hypernetwork training steps (paper: 800)

# Tasks (space-separated, required by glue_pipeline.py nargs="+")
# All 8 GLUE tasks by default
TASKS="${TASKS:-cola sst2 mrpc qqp mnli qnli rte stsb}"

# Whether to start from task-specific fine-tuned checkpoints (recommended)
USE_TASK_MODELS="${USE_TASK_MODELS:-true}"
TASK_MODEL_PREFIX="${TASK_MODEL_PREFIX:-textattack}"
LOCAL_PRETRAINED_DIR="${LOCAL_PRETRAINED_DIR:-}"  # 本地预训练模型目录，优先于 HuggingFace

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
PRETRAIN_BEFORE_COMPRESS="${PRETRAIN_BEFORE_COMPRESS:-false}"

# ── PERF_ONLY mode ────────────────────────────────────────────────────────────
# When PERF_ONLY=true: skip compression entirely, load existing checkpoints and
# only measure throughput/memory.  Useful for re-benchmarking at different
# seq_len / dtype / attn_mode without touching saved checkpoints.
#
# Example:
#   SEQ_LEN=512 DTYPE=fp32 ATTN_MODE=sdpa PERF_ONLY=true \
#     bash eval_encoder/scripts/compare_all_methods.sh
# ─────────────────────────────────────────────────────────────────────────────
PERF_ONLY="${PERF_ONLY:-false}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DTYPE="${DTYPE:-fp32}"
ATTN_MODE="${ATTN_MODE:-einsum}"      # einsum | sdpa  (naive backend only)
WARMUP_STEPS="${WARMUP_STEPS:-10}"
MEASURE_STEPS="${MEASURE_STEPS:-50}"
NUM_RUNS="${NUM_RUNS:-1}"
PERF_CSV="${PERF_CSV:-eval_encoder/eval_results/encoder_runs.csv}"

# ------------------------- derived settings -----------------------------------
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${LOG_DIR}/compare_all_${TIMESTAMP}.log"

echo "════════════════════════════════════════════════════════════════════" | tee "${SUMMARY_LOG}"
echo "Compare All Methods (GLUE)  |  ${TIMESTAMP}" | tee -a "${SUMMARY_LOG}"
echo "Repo: ${REPO_ROOT}" | tee -a "${SUMMARY_LOG}"
echo "Tasks: ${TASKS}" | tee -a "${SUMMARY_LOG}"
echo "USE_TASK_MODELS=${USE_TASK_MODELS}  TASK_MODEL_PREFIX=${TASK_MODEL_PREFIX}  LOCAL_PRETRAINED_DIR=${LOCAL_PRETRAINED_DIR}" | tee -a "${SUMMARY_LOG}"
echo "RANK=${RANK}  RANK_ATTN=${RANK_ATTN}  RANK_FFN=${RANK_FFN}  RANK_WO=${RANK_WO}" | tee -a "${SUMMARY_LOG}"
echo "QKV_MODE=${QKV_MODE}  CALIB_BATCHES=${CALIB_BATCHES}" | tee -a "${SUMMARY_LOG}"
echo "SEQ_LEN=${SEQ_LEN}  BUDGET=${BUDGET}  BACKENDS=${BACKENDS}  MODEL_ID=${MODEL_ID}" | tee -a "${SUMMARY_LOG}"
echo "TWO_STAGE=${TWO_STAGE}  PERF_ONLY=${PERF_ONLY}" | tee -a "${SUMMARY_LOG}"
if [[ "${PERF_ONLY}" == "true" ]]; then
  echo "PERF_ONLY config: bs=${BATCH_SIZE} dtype=${DTYPE} attn_mode=${ATTN_MODE} warmup=${WARMUP_STEPS} measure=${MEASURE_STEPS} csv=${PERF_CSV}" | tee -a "${SUMMARY_LOG}"
fi
echo "════════════════════════════════════════════════════════════════════" | tee -a "${SUMMARY_LOG}"
echo "" | tee -a "${SUMMARY_LOG}"

# If METHODS is provided (space-separated), use it. Otherwise use defaults.
if [[ -n "${METHODS:-}" ]]; then
  read -r -a METHODS <<< "${METHODS}"
else
  if [[ "${PRETRAIN_BEFORE_COMPRESS}" == "true" ]]; then
    METHODS=("svd" "fwsvd" "drone" "adasvd")
  else
    METHODS=("dense" "svd" "fwsvd" "drone" "adasvd")
  fi
fi

# Stages
# TWO_STAGE=true 时跑两阶段（no_finetune 看压缩后精度，with_finetune 看微调后精度）
# 普通模式：默认只跑 no_finetune（快速验证）
# pretrain_before_compress 模式：默认只跑 with_finetune（必须微调才有意义）
#   但若同时设置 TWO_STAGE=true，则也跑 no_finetune（观察压缩损失）
# Override via STAGES env var: STAGES="with_finetune" or STAGES="no_finetune with_finetune"
if [[ -n "${STAGES:-}" ]]; then
  read -ra STAGES <<< "${STAGES}"
elif [[ "${TWO_STAGE}" == "true" ]]; then
  STAGES=("no_finetune" "with_finetune")
elif [[ "${PRETRAIN_BEFORE_COMPRESS}" == "true" ]]; then
  STAGES=("with_finetune")
else
  STAGES=("no_finetune")
fi

# We'll collect JSON paths per (stage, backend, method)
declare -A JSON_BY_STAGE_BACKEND_METHOD

# Helper: derive checkpoint subdir name (mirrors glue_pipeline.py naming)
_checkpoint_subdir() {
  local method="$1"
  if [[ "${method}" == "adasvd" ]]; then
    echo "adasvd_b${BUDGET}_${QKV_MODE}_naive"
  elif [[ "${method}" == "dense" ]]; then
    echo "dense_naive"
  elif [[ -n "${RANK_ATTN}" ]] || [[ -n "${RANK_FFN}" ]] || [[ -n "${RANK_WO}" ]]; then
    local ra="${RANK_ATTN:-${RANK}}"
    local rf="${RANK_FFN:-${RANK}}"
    local rw="${RANK_WO:-${RANK}}"
    echo "${method}_ra${ra}_rf${rf}_rw${rw}_${QKV_MODE}_naive"
  else
    echo "${method}_r${RANK}_${QKV_MODE}_naive"
  fi
}

# Helper: run performance-only measurement on an existing checkpoint (PERF_ONLY=true)
_run_perf_only() {
  local method="$1"
  local task="$2"
  local backend="$3"

  local subdir
  subdir="$(_checkpoint_subdir "${method}")"

  local model_dir
  if [[ "${USE_TASK_MODELS}" == "true" || -n "${LOCAL_PRETRAINED_DIR}" ]]; then
    model_dir="${REPO_ROOT}/eval_encoder/models/${task}/${subdir}"
  else
    model_dir="${REPO_ROOT}/eval_encoder/models/${subdir}"
  fi

  if [[ ! -d "${model_dir}" ]]; then
    echo "[perf_only][warn] Checkpoint not found: ${model_dir} — skipping" | tee -a "${SUMMARY_LOG}"
    return 0
  fi

  echo "[perf_only] method=${method} task=${task} backend=${backend} attn_mode=${ATTN_MODE}" | tee -a "${SUMMARY_LOG}"
  echo "[perf_only] checkpoint: ${model_dir}" | tee -a "${SUMMARY_LOG}"

  local cmd=(
    python eval_encoder/run_encoder_benchmark.py
    --load_model_dir "${model_dir}"
    --method "${method}"
    --task "${task}"
    --backend "${backend}"
    --attn_mode "${ATTN_MODE}"
    --seq_len "${SEQ_LEN}"
    --batch_size "${BATCH_SIZE}"
    --dtype "${DTYPE}"
    --skip_eval
    --warmup_steps "${WARMUP_STEPS}"
    --measure_steps "${MEASURE_STEPS}"
    --num_runs "${NUM_RUNS}"
    --out_csv "${PERF_CSV}"
  )

  echo "CMD: ${cmd[*]}" | tee -a "${SUMMARY_LOG}"
  "${cmd[@]}" 2>&1 | tee -a "${SUMMARY_LOG}"
}

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

  # ── PERF_ONLY mode: load existing checkpoint, skip compression/finetune ──
  if [[ "${PERF_ONLY}" == "true" ]]; then
    local task perf_ok=0 perf_fail=0
    for task in ${TASKS}; do
      if _run_perf_only "${method}" "${task}" "${backend}"; then
        echo "✅ perf_only done: method=${method} task=${task}" | tee -a "${SUMMARY_LOG}"
        perf_ok=$((perf_ok + 1))
      else
        echo "❌ perf_only failed: method=${method} task=${task}" | tee -a "${SUMMARY_LOG}"
        perf_fail=$((perf_fail + 1))
      fi
    done
    echo "" | tee -a "${SUMMARY_LOG}"
    [[ "${perf_fail}" -eq 0 ]]  # propagate failure to caller
    return
  fi

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
  if [[ -n "${LOCAL_PRETRAINED_DIR}" ]]; then
    env_prefix+=("LOCAL_PRETRAINED_DIR=${LOCAL_PRETRAINED_DIR}")
  fi
  env_prefix+=("SKIP_FINETUNING=${skip_finetuning}")
  env_prefix+=("SKIP_COMPRESSION=false")
  env_prefix+=("REUSE_CHECKPOINT=${reuse_checkpoint}")
  env_prefix+=("PRETRAIN_BEFORE_COMPRESS=${PRETRAIN_BEFORE_COMPRESS}")
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

  # Add QKV mode, seq_len and calibration batches
  env_prefix+=("QKV_MODE=${QKV_MODE}")
  env_prefix+=("SEQ_LEN=${SEQ_LEN}")
  env_prefix+=("CALIB_BATCHES=${CALIB_BATCHES}")

  if [[ "${method}" == "adasvd" ]]; then
    env_prefix+=("BUDGET=${BUDGET}")
    env_prefix+=("ADASVD_CALIB_SAMPLES=${ADASVD_CALIB_SAMPLES}")
    env_prefix+=("ADASVD_STEPS=${ADASVD_STEPS}")
  elif [[ "${method}" == "svd" || "${method}" == "fwsvd" || "${method}" == "drone" ]]; then
    env_prefix+=("RANK=${RANK}")
  fi

  # Run
  local start_ts end_ts dur run_exit
  start_ts=$(date +%s)
  (
    export "${env_prefix[@]}"
    bash eval_encoder/scripts/one_click_glue.sh
  ) 2>&1 | tee -a "${SUMMARY_LOG}"
  run_exit="${PIPESTATUS[0]}"
  end_ts=$(date +%s)
  dur=$((end_ts - start_ts))
  if [[ "${run_exit}" -ne 0 ]]; then
    echo "❌ FAILED: stage=${stage} backend=${backend} method=${method} (exit=${run_exit}, ${dur}s)" | tee -a "${SUMMARY_LOG}"
    return "${run_exit}"
  fi
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
    "cola": "CoLA",
    "sst2": "SST-2",
    "mrpc": "MRPC",
    "qqp":  "QQP",
    "mnli": "MNLI",
    "qnli": "QNLI",
    "rte":  "RTE",
    "stsb": "STS-B",
}

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _get(res, key, fallback_keys=()):
    """Get a float value from a result dict, trying fallback keys if needed."""
    if not res:
        return None
    v = res.get(key)
    if v is None:
        for fk in fallback_keys:
            v = res.get(fk)
            if v is not None:
                break
    if v is None:
        # last resort: use final of primary_metric
        metrics = res.get("metrics", {})
        pm = metrics.get("primary_metric")
        if pm and "final" in metrics and pm in metrics.get("final", {}):
            return float(metrics["final"][pm])
        return None
    return float(v)

def stage_backend_maps(mp):
    out = {}
    for k, v in mp.items():
        parts = k.split(":")
        if len(parts) == 3:
            stage, backend, method = parts
            out.setdefault(stage, {}).setdefault(backend, {})[method] = v
    return out

def fmt(v):
    return "-    " if v is None else f"{v:.4f}"

def print_table(stage, mm):
    """Print one table per stage showing both naive (N) and flashsvd (F) accuracy."""
    methods_order = ["dense","svd","fwsvd","drone","adasvd"]
    W = 10 + len(tasks_order) * 14 + 20
    print("=" * W)
    print(f"Stage: {stage}   (N = naive backend, F = flashsvd backend)")
    print("=" * W)
    # Header
    hdr = f"{'Method':<9}"
    for t in tasks_order:
        hdr += f"  {task_hdr[t]:>11}"
    hdr += f"  {'G-Avg':>6}  {'A-Avg':>6}"
    print(hdr)
    print(f"{'':9}" + "".join(f"  {'N':>5} {'F':>5}" for _ in tasks_order) +
          f"  {'N':>3} {'F':>3}  {'N':>3} {'F':>3}")
    print("-" * W)

    # Collect all paths from any backend key in mm (we pick the first available)
    method_paths = {}
    for backend_dict in mm.values():
        for method, path in backend_dict.items():
            if method not in method_paths:
                method_paths[method] = path

    for m in methods_order:
        path = method_paths.get(m)
        if not path or not os.path.exists(path):
            print(f"{m:<9}  (no result)")
            continue
        obj = load(path)
        results = {r.get("task"): r for r in obj.get("results", [])}
        row = f"{m:<9}"
        for t in tasks_order:
            res = results.get(t)
            # Naive: best_value (primary); FlashSVD: best_value_flashsvd
            vn = _get(res, "best_value")
            vf = _get(res, "best_value_flashsvd")
            row += f"  {fmt(vn)} {fmt(vf)}"
        summ = obj.get("summary", {})
        gn = float(summ.get("G-Avg", {}).get("final", 0.0))
        an = float(summ.get("A-Avg", {}).get("final", 0.0))
        # FlashSVD G-Avg / A-Avg: compute from per-task flashsvd values
        gf_vals, af_vals = [], []
        for t in tasks_order:
            res = results.get(t)
            vf = _get(res, "best_value_flashsvd")
            if vf is not None:
                metric = (res.get("best_metric") or
                          res.get("metrics", {}).get("primary_metric", ""))
                norm = (vf + 1) / 2 if metric in ("matthews_correlation", "pearson") else vf
                gf_vals.append(norm)
                if metric == "accuracy":
                    af_vals.append(vf)
        gf = sum(gf_vals) / len(gf_vals) if gf_vals else None
        af = sum(af_vals) / len(af_vals) if af_vals else None
        row += f"  {gn:.3f} {fmt(gf)[:5]}  {an:.3f} {fmt(af)[:5]}"
        print(row)
    print()

nested = stage_backend_maps(mp)
# Merge all backends into one dict per stage (we show N vs F from the result JSON itself)
merged = {}
for stage, bd in nested.items():
    merged[stage] = bd

for stage in sorted(merged.keys()):
    print_table(stage, merged[stage])
PY

echo "Log saved to: ${SUMMARY_LOG}"
echo "Success: ${SUCCESS}  Fail: ${FAIL}"
if [[ "${FAIL}" -gt 0 ]]; then
  exit 1
fi
exit 0
