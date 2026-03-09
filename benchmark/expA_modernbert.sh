#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# expA_modernbert.sh  —  Quality Experiment for ModernBERT
#
# Full pipeline equivalent to expA.sh but for ModernBERT-base:
#   Phase 1: GLUE (8 tasks, 4 methods, stage1 + stage2)
#             base → task fine-tune → compress → eval → fine-tune
#   Phase 2: SuperGLUE Stage1 only
#     2a. Task-specific (boolq, rte_sg, wic, cb):
#             base → task fine-tune → compress → eval
#     2b. NLI-based (copa, hans, anli_r1/r2/r3):
#             use MNLI pretrain checkpoint → compress → eval
#             (hans/anli: calib_task=mnli since they have no train split)
#   Phase 3: SuperGLUE fine-tune (boolq + wic, stage1 + stage2)
#             base → task fine-tune → compress → eval → fine-tune
#
# Usage
# ─────────────────────────────────────────────────────────────────────────────
#   cd lowrankarena/
#   bash benchmark/expA_modernbert.sh                   # full run
#   PHASES=glue bash benchmark/expA_modernbert.sh       # GLUE only
#
# Overridable variables
#   MODEL_ID      (default: answerdotai/ModernBERT-base)
#   PHASES        (default: "glue superglue superglue_finetune")
#   METHODS       (default: "svd fwsvd drone adasvd")
#   TASKS_GLUE, TASKS_SUPERGLUE_TASKSPECIFIC, TASKS_SUPERGLUE_NLI,
#   TASKS_SUPERGLUE_FINETUNE
#   TWO_STAGE, RECOMPRESS, OUT_CSV
#   All canonical config vars (QKV_MODE, RANK_ATTN, RANK_FFN, RANK_WO, etc.)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.."; pwd)"
cd "${REPO_ROOT}"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-experiments/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/expA_modernbert_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[log] → ${LOG_FILE}"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID="${MODEL_ID:-answerdotai/ModernBERT-base}"
# Slug used for CSV naming: strip HF prefix (answerdotai/) and lowercase
_MODEL_SLUG="${MODEL_ID##*/}"
_MODEL_SLUG="${_MODEL_SLUG,,}"   # lowercase

# ── Canonical config ──────────────────────────────────────────────────────────
QKV_MODE="${QKV_MODE:-per_head}"
if [[ -z "${RANK_ATTN:-}" ]]; then
    [[ "${QKV_MODE}" == "full" ]] && RANK_ATTN=312 || RANK_ATTN=48
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

# ── Overridable ───────────────────────────────────────────────────────────────
PHASES="${PHASES:-glue superglue superglue_finetune}"
METHODS="${METHODS:-svd fwsvd drone adasvd}"
TWO_STAGE="${TWO_STAGE:-true}"
RECOMPRESS="${RECOMPRESS:-false}"
TASKS_GLUE="${TASKS_GLUE:-cola sst2 mrpc qqp mnli qnli rte stsb}"
# Phase 2a: tasks that have their own train split → per-task pretrain
TASKS_SUPERGLUE_TASKSPECIFIC="${TASKS_SUPERGLUE_TASKSPECIFIC:-boolq rte_sg wic cb}"
# Phase 2b: NLI-based tasks → use MNLI pretrained checkpoint as base
TASKS_SUPERGLUE_NLI="${TASKS_SUPERGLUE_NLI:-copa hans anli_r1 anli_r2 anli_r3}"
TASKS_SUPERGLUE_FINETUNE="${TASKS_SUPERGLUE_FINETUNE:-boolq wic}"

# Model base dir (always modernbert)
MODEL_BASE_DIR="${MODEL_BASE_DIR:-compressed_models/modernbert}"
# MNLI pretrain checkpoint (built during Phase 1; reused in Phase 2b)
MNLI_PRETRAIN_DIR="${MNLI_PRETRAIN_DIR:-${MODEL_BASE_DIR}/dense/mnli/pretrained_base}"

# Output CSV: auto-named by model slug (e.g. expA_modernbert-base.csv)
OUT_CSV="${OUT_CSV:-experiments/results/expA_${_MODEL_SLUG}.csv}"

echo "══════════════════════════════════════════════════════════════════════"
echo "  expA_modernbert — ModernBERT Quality Experiment"
echo "  model_id:  ${MODEL_ID}"
echo "  phases:    ${PHASES}"
echo "  methods:   ${METHODS}"
echo "  config:    qkv=${QKV_MODE} ra${RANK_ATTN}_rf${RANK_FFN}_rw${RANK_WO}  budget=${BUDGET}"
echo "  dtype:     ${DTYPE}   seq_len=${SEQ_LEN}   bs=${BATCH_SIZE}"
echo "  two_stage: ${TWO_STAGE}   recompress: ${RECOMPRESS}"
echo "  out_csv:   ${OUT_CSV}"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

OK=0; FAIL=0
_PHASES=" ${PHASES} "

# ── Common env for all compare_all_methods.sh calls ──────────────────────────
_COMMON_ENV=(
    MODEL_ID="${MODEL_ID}"
    MODEL_BASE_DIR="${MODEL_BASE_DIR}"
    METHODS="${METHODS}"
    BACKENDS="${BACKENDS:-naive}"
    QKV_MODE="${QKV_MODE}"
    RANK_ATTN="${RANK_ATTN}"
    RANK_FFN="${RANK_FFN}"
    RANK_WO="${RANK_WO}"
    BUDGET="${BUDGET}"
    DTYPE="${DTYPE}"
    SEQ_LEN="${SEQ_LEN}"
    BATCH_SIZE="${BATCH_SIZE}"
    CALIB_BATCHES="${CALIB_BATCHES}"
    ADASVD_CALIB_SAMPLES="${ADASVD_CALIB_SAMPLES}"
    ADASVD_STEPS="${ADASVD_STEPS}"
    AUTO_FIGURES="false"
    PERF_CSV="${OUT_CSV}"
    REUSE_CHECKPOINT="$( [[ "${RECOMPRESS}" == "true" ]] && echo "false" || echo "true" )"
)

# ── Phase 1: GLUE ─────────────────────────────────────────────────────────────
if [[ "${_PHASES}" == *" glue "* ]]; then
    echo "══ Phase 1: GLUE ══════════════════════════════════════════════════"
    echo "   tasks:   ${TASKS_GLUE}"
    echo "   stages:  $([ "${TWO_STAGE}" = "true" ] && echo "no_finetune + with_finetune" || echo "no_finetune only")"
    echo ""

    env "${_COMMON_ENV[@]}" \
        TASKS="${TASKS_GLUE}" \
        TWO_STAGE="${TWO_STAGE}" \
        PRETRAIN_BEFORE_COMPRESS="true" \
        bash benchmark/compare_all_methods.sh \
    && OK=$((OK + 1)) || FAIL=$((FAIL + 1))
    echo ""
fi

# ── Phase 2a: SuperGLUE Stage1 — task-specific pretrain ───────────────────────
if [[ "${_PHASES}" == *" superglue "* ]]; then
    echo "══ Phase 2a: SuperGLUE Stage1 (task-specific pretrain) ════════════"
    echo "   tasks:  ${TASKS_SUPERGLUE_TASKSPECIFIC}"
    echo ""

    env "${_COMMON_ENV[@]}" \
        TASKS="${TASKS_SUPERGLUE_TASKSPECIFIC}" \
        TWO_STAGE="false" \
        PRETRAIN_BEFORE_COMPRESS="true" \
        bash benchmark/compare_all_methods.sh \
    && OK=$((OK + 1)) || FAIL=$((FAIL + 1))
    echo ""

    # ── Phase 2b: SuperGLUE Stage1 — NLI-based tasks (copa/hans/anli) ─────────
    echo "══ Phase 2b: SuperGLUE Stage1 (NLI-based, uses MNLI pretrain) ═════"
    echo "   tasks:            ${TASKS_SUPERGLUE_NLI}"
    echo "   mnli_pretrain_dir: ${MNLI_PRETRAIN_DIR}"
    echo ""

    if [[ ! -d "${MNLI_PRETRAIN_DIR}" ]]; then
        echo "[warn] MNLI pretrain checkpoint not found: ${MNLI_PRETRAIN_DIR}"
        echo "[warn] Run Phase 1 first (includes MNLI fine-tuning), or set MNLI_PRETRAIN_DIR."
        echo "[warn] Skipping Phase 2b."
        FAIL=$((FAIL + 1))
    else
        # hans and anli have no train split — force calib_task=mnli for fwsvd/drone/adasvd
        # copa has copa train data but still benefits from MNLI base
        env "${_COMMON_ENV[@]}" \
            TASKS="${TASKS_SUPERGLUE_NLI}" \
            TWO_STAGE="false" \
            PRETRAIN_BEFORE_COMPRESS="false" \
            LOCAL_PRETRAINED_DIR="${MNLI_PRETRAIN_DIR}" \
            CALIB_TASK="mnli" \
            bash benchmark/compare_all_methods.sh \
        && OK=$((OK + 1)) || FAIL=$((FAIL + 1))
        echo ""
    fi
fi

# ── Phase 3: SuperGLUE fine-tune (BoolQ + WiC) ────────────────────────────────
if [[ "${_PHASES}" == *" superglue_finetune "* ]]; then
    echo "══ Phase 3: SuperGLUE fine-tune (BoolQ + WiC) ══════════════════════"
    echo "   tasks:  ${TASKS_SUPERGLUE_FINETUNE}"
    echo "   stages: compress + fine-tune (TWO_STAGE=true)"
    echo ""

    env "${_COMMON_ENV[@]}" \
        TASKS="${TASKS_SUPERGLUE_FINETUNE}" \
        TWO_STAGE="true" \
        PRETRAIN_BEFORE_COMPRESS="true" \
        bash benchmark/compare_all_methods.sh \
    && OK=$((OK + 1)) || FAIL=$((FAIL + 1))
    echo ""
fi

echo "══════════════════════════════════════════════════════════════════════"
echo "  expA_modernbert done  success=${OK}  failed=${FAIL}"
echo ""
echo "  Output CSV: ${OUT_CSV}"
echo "══════════════════════════════════════════════════════════════════════"

[[ "${FAIL}" -eq 0 ]]
