#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# expA.sh  —  Quality Experiment (canonical paper reproduction entrypoint)
#
# Fixed canonical config, one-click reproduction of all quality experiments:
#   Phase 1: GLUE (8 tasks, 4 methods, stage1 + stage2)
#   Phase 2: SuperGLUE + HANS + ANLI (7 tasks, 4 methods, naive backend)
#
# Canonical config (the single authoritative parameter set for paper results)
# ────────────────────────────────────────────────────────────────────────────
#   QKV_MODE  = per_head
#   RANK_ATTN = 48   RANK_FFN = 256   RANK_WO = 208
#   BUDGET    = 0.527   (AdaSVD, parameter count equivalent to the ranks above)
#   DTYPE     = bf16
#   SEQ_LEN   = 512    BATCH_SIZE = 32
#
# Usage
# ────────────────────────────────────────────────────────────────────────────
#   cd lowrankarena/
#   bash benchmark/expA.sh                # full run (GLUE + SuperGLUE)
#   PHASES=glue bash benchmark/expA.sh    # GLUE only
#   PHASES=superglue bash benchmark/expA.sh  # SuperGLUE only
#
# Overridable variables
#   PHASES="glue superglue"           # phase subset
#   METHODS="svd fwsvd"               # method subset (default: 4 methods)
#   TASKS_GLUE="mnli stsb"            # GLUE task subset
#   TASKS_SUPERGLUE="boolq hans"      # SuperGLUE task subset
#   TWO_STAGE=false                   # run stage1 only (no_finetune), skip fine-tuning
#   RECOMPRESS=true                   # force recompression (ignore existing checkpoints)
#   OUT_CSV=path/to.csv               # path for all CSV results (shared by GLUE + SuperGLUE)
#
# Output
#   GLUE JSON : experiments/glue/glue_results_{method}_*.json
#               → summarized to CSV via benchmark/analysis/collect_glue_results.py
#   ALL CSV   : experiments/results/expA.csv (shared by GLUE + SuperGLUE)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-experiments/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/expA_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[log] → ${LOG_FILE}"

# ── Canonical config (per_head is the paper standard; override via env vars to run full-matrix) ──
#   per_head (default):  QKV_MODE=per_head  RANK_ATTN=48   param_ratio≈0.527
#   full-matrix:         QKV_MODE=full       RANK_ATTN=312  param_ratio≈0.527
#
# Note: no export used — all canonical variables are computed locally in this script
#   and passed explicitly via command prefixes to sub-scripts.
#   This avoids stale environment variables from previous runs (e.g. TASKS_GLUE=stsb)
#   polluting the current experiment.
QKV_MODE="${QKV_MODE:-per_head}"
if [[ -z "${RANK_ATTN:-}" ]]; then
    [[ "${QKV_MODE}" == "full" ]] && RANK_ATTN=312 || RANK_ATTN=48
else
    RANK_ATTN="${RANK_ATTN}"
fi
RANK_FFN="${RANK_FFN:-256}"
RANK_WO="${RANK_WO:-208}"
BUDGET="${BUDGET:-0.527}"
DTYPE="${DTYPE:-bf16}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CALIB_BATCHES="${CALIB_BATCHES:-16}"
ADASVD_CALIB_SAMPLES="${ADASVD_CALIB_SAMPLES:-4000}"
ADASVD_STEPS="${ADASVD_STEPS:-800}"

# ── Overridable configuration ──────────────────────────────────────────────────
PHASES="${PHASES:-glue superglue}"
METHODS="${METHODS:-svd fwsvd drone adasvd}"
TASKS_GLUE="${TASKS_GLUE:-cola sst2 mrpc qqp mnli qnli rte stsb}"
TASKS_SUPERGLUE="${TASKS_SUPERGLUE:-boolq rte_sg wic copa cb hans anli_r1 anli_r2 anli_r3}"
TWO_STAGE="${TWO_STAGE:-true}"
RECOMPRESS="${RECOMPRESS:-false}"
MODEL_BASE_DIR="${MODEL_BASE_DIR:-compressed_models/bert}"
OUT_CSV="${OUT_CSV:-experiments/results/expA.csv}"

echo "══════════════════════════════════════════════════════════════════════"
echo "  expA — Quality Experiment"
echo "  phases:    ${PHASES}"
echo "  methods:   ${METHODS}"
echo "  config:    qkv=${QKV_MODE} ra${RANK_ATTN}_rf${RANK_FFN}_rw${RANK_WO}  budget=${BUDGET}"
echo "  dtype:     ${DTYPE}   seq_len=${SEQ_LEN}   bs=${BATCH_SIZE}"
echo "  two_stage: ${TWO_STAGE}   recompress: ${RECOMPRESS}"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

OK=0
FAIL=0

# ── Phase 1: GLUE ─────────────────────────────────────────────────────────────
if [[ "${PHASES}" == *"glue"* ]]; then
    echo "══ Phase 1: GLUE ══════════════════════════════════════════════════"
    echo "   tasks:  ${TASKS_GLUE}"
    echo "   stages: $([ "${TWO_STAGE}" = "true" ] && echo "no_finetune + with_finetune" || echo "no_finetune only")"
    echo ""

    TASKS="${TASKS_GLUE}" \
    METHODS="${METHODS}" \
    TWO_STAGE="${TWO_STAGE}" \
    BACKENDS="naive" \
    USE_TASK_MODELS="true" \
    TASK_MODEL_PREFIX="textattack" \
    PRETRAIN_BEFORE_COMPRESS="false" \
    AUTO_FIGURES="false" \
    PERF_CSV="${OUT_CSV}" \
    QKV_MODE="${QKV_MODE}" \
    RANK_ATTN="${RANK_ATTN}" \
    RANK_FFN="${RANK_FFN}" \
    RANK_WO="${RANK_WO}" \
    BUDGET="${BUDGET}" \
    DTYPE="${DTYPE}" \
    SEQ_LEN="${SEQ_LEN}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    CALIB_BATCHES="${CALIB_BATCHES}" \
    ADASVD_CALIB_SAMPLES="${ADASVD_CALIB_SAMPLES}" \
    ADASVD_STEPS="${ADASVD_STEPS}" \
    bash benchmark/compare_all_methods.sh \
    && OK=$((OK + 1)) || FAIL=$((FAIL + 1))

    echo ""
fi

# ── Phase 2: SuperGLUE + HANS + ANLI ─────────────────────────────────────────
if [[ "${PHASES}" == *"superglue"* ]]; then
    echo "══ Phase 2: SuperGLUE / HANS / ANLI ══════════════════════════════"
    echo "   tasks: ${TASKS_SUPERGLUE}"
    echo ""

    TASKS="${TASKS_SUPERGLUE}" \
    METHODS="${METHODS}" \
    BACKENDS="naive" \
    RECOMPRESS="${RECOMPRESS}" \
    MODEL_BASE_DIR="${MODEL_BASE_DIR}" \
    OUT_CSV="${OUT_CSV}" \
    QKV_MODE="${QKV_MODE}" \
    RANK_ATTN="${RANK_ATTN}" \
    RANK_FFN="${RANK_FFN}" \
    RANK_WO="${RANK_WO}" \
    BUDGET="${BUDGET}" \
    DTYPE="${DTYPE}" \
    SEQ_LEN="${SEQ_LEN}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    CALIB_BATCHES="${CALIB_BATCHES}" \
    ADASVD_CALIB_SAMPLES="${ADASVD_CALIB_SAMPLES}" \
    ADASVD_STEPS="${ADASVD_STEPS}" \
    bash benchmark/run_superglue_benchmark.sh \
    && OK=$((OK + 1)) || FAIL=$((FAIL + 1))

    echo ""
fi

echo "══════════════════════════════════════════════════════════════════════"
echo "  expA done  success=${OK}  failed=${FAIL}"
echo ""
echo "  Output:"
echo "    GLUE JSON → experiments/glue/glue_results_*.json"
echo "    ALL CSV   → ${OUT_CSV}"
echo ""
echo "  Summary figures:"
echo "    python experiments/collect_glue_results.py"
echo "    python experiments/gen_figures.py"
echo "══════════════════════════════════════════════════════════════════════"

[[ "${FAIL}" -eq 0 ]]
