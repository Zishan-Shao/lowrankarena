#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# expE.sh  —  Supplementary Experiments (E-1 through E-3b)
#
# Four phases of supplementary experiments for the paper:
#
#   E-1  Timing boundary documentation (no new experiments; prints statement)
#   E-2  Logit alignment check (flashsvd vs naive, per backend)
#   E-3a Per-step training time (naive / sdpa only — full fwd+bwd+opt step)
#   E-3b Fine-tune recovery curve (accuracy vs training step count)
#
# Design notes
# ────────────────────────────────────────────────────────────────────────────
#   • E-2 delegates to expB.sh with ALIGN=1 — reuses the same CSV schema
#   • E-3a/E-3b use new scripts (run_train_timing.py / run_recovery_curve.py)
#   • flashsvd / flashsvd15 Triton kernels have NO autograd → never included
#     in E-3a training steps.  If user passes them in E3A_BACKENDS, the
#     script prints SKIP (exit 2) and the wrapper counts it as SKIP, not FAIL.
#   • Same OK/FAIL/SKIP counter pattern as expB/expC
#   • No interactive prompts (fully batch-safe)
#
# Usage
# ────────────────────────────────────────────────────────────────────────────
#   cd lowrankarena/
#   bash benchmark/expE.sh                        # all four phases
#   PHASES="e2"   bash benchmark/expE.sh           # E-2 only
#   PHASES="e3a"  TASKS="mnli" bash benchmark/expE.sh
#   PHASES="e3b"  TASKS="mnli" METHODS="svd" bash benchmark/expE.sh
#
# Overridable env vars
#   PHASES="e1 e2 e3a e3b"
#   TASKS="mnli mrpc"            # for E-3a and E-3b
#   E2_TASKS="mnli"              # tasks for E-2 alignment (default: mnli)
#   METHODS="svd fwsvd drone"    # for E-3a and E-3b
#   E3A_BACKENDS="naive sdpa"    # flashsvd* → auto-skipped (no autograd)
#   E3B_EVAL_STEPS="0 200 500 1000"
#   E3B_NUM_EPOCHS=3
#   E3B_EVAL_BACKENDS="naive"    # comma-sep; add flashsvd15 for eval-only rows
#   WARMUP=50  MEASURE=100       # E-3a timing (more steps = lower variance)
#   DTYPE=bf16
#   SEQ_LEN=512
#   BATCH_SIZE=32
#   LR=2e-5
#   RANK_ATTN=48  RANK_FFN=256  RANK_WO=208  QKV_MODE=per_head
#   BUDGET=0.527
#   MODEL_BASE_DIR=compressed_models/bert
#   OUT_EXPB=experiments/results/expE_alignment.csv
#   OUT_E3A=experiments/results/expE_train_timing.csv
#   OUT_E3B=experiments/results/expE_recovery.csv
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-experiments/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/expE_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[log] → ${LOG_FILE}"

# ── Configuration ─────────────────────────────────────────────────────────────
PHASES="${PHASES:-e1 e2 e3a e3b}"

# Canonical config (matches expA)
RANK_ATTN="${RANK_ATTN:-48}"
RANK_FFN="${RANK_FFN:-256}"
RANK_WO="${RANK_WO:-208}"
QKV_MODE="${QKV_MODE:-per_head}"
BUDGET="${BUDGET:-0.527}"
DTYPE="${DTYPE:-bf16}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# E-3a/E-3b shared
TASKS="${TASKS:-mnli mrpc}"
METHODS="${METHODS:-svd fwsvd drone}"
LR="${LR:-2e-5}"
MODEL_BASE_DIR="${MODEL_BASE_DIR:-compressed_models/bert}"

# E-2
E2_TASKS="${E2_TASKS:-mnli}"
E2_BACKENDS="${E2_BACKENDS:-naive sdpa flashsvd flashsvd15}"
E2_INPUT_MODES="${E2_INPUT_MODES:-real}"
E2_REPEAT="${E2_REPEAT:-1}"
OUT_EXPB="${OUT_EXPB:-experiments/results/expE_alignment.csv}"

# E-3a
E3A_BACKENDS="${E3A_BACKENDS:-naive sdpa}"
WARMUP="${WARMUP:-50}"
MEASURE="${MEASURE:-100}"
OUT_E3A="${OUT_E3A:-experiments/results/expE_train_timing.csv}"

# E-3b
E3B_EVAL_STEPS="${E3B_EVAL_STEPS:-0 200 500 1000}"
E3B_NUM_EPOCHS="${E3B_NUM_EPOCHS:-3}"
E3B_EVAL_BACKENDS="${E3B_EVAL_BACKENDS:-naive}"
OUT_E3B="${OUT_E3B:-experiments/results/expE_recovery.csv}"

# ── Checkpoint subdirectory name ──────────────────────────────────────────────
_model_subdir() {
    local method="$1"
    if [[ "${method}" == "adasvd" ]]; then
        echo "adasvd_b${BUDGET}_${QKV_MODE}_naive"
    else
        echo "${method}_ra${RANK_ATTN}_rf${RANK_FFN}_rw${RANK_WO}_${QKV_MODE}_naive"
    fi
}

# ── Banner ────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
echo "  expE — Supplementary Experiments"
echo "  phases:      ${PHASES}"
echo "  tasks:       ${TASKS}   methods: ${METHODS}"
echo "  dtype:       ${DTYPE}   seq_len: ${SEQ_LEN}   bs: ${BATCH_SIZE}"
echo "  config:      ra${RANK_ATTN}_rf${RANK_FFN}_rw${RANK_WO}  qkv=${QKV_MODE}  budget=${BUDGET}"
echo "  models:      ${MODEL_BASE_DIR}"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

OK=0
FAIL=0
SKIP=0

# ═══════════════════════════════════════════════════════════════════════════════
# Phase E-1: Timing Boundary Documentation (no experiments)
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "${PHASES}" == *"e1"* ]]; then
    echo "══ Phase E-1: Timing Boundary Documentation ══════════════════════"
    echo ""
    echo "  Paper statement (Section: Experimental Setup):"
    echo ""
    echo "  All inference latency / throughput numbers reported in expB/expC"
    echo "  measure GPU-side forward pass only.  The timing boundary is:"
    echo ""
    echo "    START: immediately before model(**batch) on GPU"
    echo "    STOP:  after torch.cuda.synchronize() returns"
    echo ""
    echo "  Excluded from timing:"
    echo "    • Host-to-Device (H2D) data transfer"
    echo "    • CPU preprocessing (tokenization, collation)"
    echo "    • Device-to-Host (D2H) logit copy"
    echo "    • Optimizer step (training pipeline)"
    echo ""
    echo "  This boundary is identical across all backends (naive / sdpa /"
    echo "  flashsvd / flashsvd15), ensuring apples-to-apples comparison."
    echo ""
    echo "  Contrast with E-3a (full training step = fwd + bwd + opt):"
    echo "    START: before model(**batch)   STOP: after optimizer.step()"
    echo ""
    echo "  [E-1] Done — no experiments run (documentation only)"
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase E-2: Logit Alignment Check (flashsvd vs naive)
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "${PHASES}" == *"e2"* ]]; then
    echo "══ Phase E-2: Logit Alignment Check ══════════════════════════════"
    echo "   Delegates to expB.sh with ALIGN=1"
    echo "   tasks:        ${E2_TASKS}"
    echo "   backends:     ${E2_BACKENDS}"
    echo "   input_modes:  ${E2_INPUT_MODES}"
    echo "   repeat:       ${E2_REPEAT}"
    echo "   out_csv:      ${OUT_EXPB}  (separate file, does not pollute expB.csv)"
    echo ""

    rc=0
    ALIGN=1 \
    TASKS="${E2_TASKS}" \
    METHODS="${METHODS:-svd}" \
    BACKENDS="${E2_BACKENDS}" \
    INPUT_MODES="${E2_INPUT_MODES}" \
    REPEAT="${E2_REPEAT}" \
    DTYPE="${DTYPE}" \
    SEQ_LEN="${SEQ_LEN}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    WARMUP="${WARMUP:-10}" \
    MEASURE="${MEASURE:-50}" \
    RANK_ATTN="${RANK_ATTN}" \
    RANK_FFN="${RANK_FFN}" \
    RANK_WO="${RANK_WO}" \
    QKV_MODE="${QKV_MODE}" \
    BUDGET="${BUDGET}" \
    MODEL_BASE_DIR="${MODEL_BASE_DIR}" \
    OUT_CSV="${OUT_EXPB}" \
    bash benchmark/expB.sh || rc=$?

    if   [[ $rc -eq 0 ]]; then OK=$((OK + 1)); echo "[E-2] OK"
    elif [[ $rc -eq 2 ]]; then SKIP=$((SKIP + 1)); echo "[E-2] SKIP"
    else FAIL=$((FAIL + 1)); echo "[E-2] FAILED (expB.sh exit=$rc)"; fi
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase E-3a: Per-Step Training Time
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "${PHASES}" == *"e3a"* ]]; then
    echo "══ Phase E-3a: Per-Step Training Time ════════════════════════════"
    echo "   tasks:    ${TASKS}"
    echo "   methods:  ${METHODS}"
    echo "   backends: ${E3A_BACKENDS}  (flashsvd* → auto-SKIP: no autograd)"
    echo "   warmup:   ${WARMUP}   measure: ${MEASURE}"
    echo "   out_csv:  ${OUT_E3A}"
    echo ""

    mkdir -p "$(dirname "${OUT_E3A}")"

    for TASK in ${TASKS}; do
        for METHOD in ${METHODS}; do
            SUBDIR="$(_model_subdir "${METHOD}")"
            MODEL_DIR="${MODEL_BASE_DIR}/${TASK}/${SUBDIR}"

            if [[ ! -d "${MODEL_DIR}" ]]; then
                echo "   [skip] Checkpoint not found: ${MODEL_DIR}"
                SKIP=$((SKIP + 1))
                continue
            fi

            echo "  ── ${TASK} / ${METHOD}  (${MODEL_DIR})"

            for BACKEND in ${E3A_BACKENDS}; do
                echo "     → backend=${BACKEND}"
                rc=0
                python benchmark/tools/run_train_timing.py \
                    --model_dir  "${MODEL_DIR}" \
                    --task       "${TASK}" \
                    --backend    "${BACKEND}" \
                    --seq_len    "${SEQ_LEN}" \
                    --batch_size "${BATCH_SIZE}" \
                    --dtype      "${DTYPE}" \
                    --warmup     "${WARMUP}" \
                    --measure    "${MEASURE}" \
                    --out_csv    "${OUT_E3A}" \
                || rc=$?

                if   [[ $rc -eq 0 ]]; then OK=$((OK + 1))
                elif [[ $rc -eq 2 ]]; then SKIP=$((SKIP + 1)); echo "     [SKIP] ${TASK}/${METHOD}/${BACKEND} (no autograd)"
                else FAIL=$((FAIL + 1)); echo "     [FAILED] ${TASK}/${METHOD}/${BACKEND} (exit=${rc})"; fi
            done
            echo ""
        done
    done

    echo "   E-3a out → ${OUT_E3A}"
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase E-3b: Fine-tune Recovery Curve
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "${PHASES}" == *"e3b"* ]]; then
    echo "══ Phase E-3b: Fine-tune Recovery Curve ══════════════════════════"
    echo "   tasks:         ${TASKS}"
    echo "   methods:       ${METHODS}"
    echo "   eval_steps:    ${E3B_EVAL_STEPS}"
    echo "   num_epochs:    ${E3B_NUM_EPOCHS}"
    echo "   eval_backends: ${E3B_EVAL_BACKENDS}"
    echo "   out_csv:       ${OUT_E3B}"
    echo ""

    mkdir -p "$(dirname "${OUT_E3B}")"

    for TASK in ${TASKS}; do
        for METHOD in ${METHODS}; do
            SUBDIR="$(_model_subdir "${METHOD}")"
            MODEL_DIR="${MODEL_BASE_DIR}/${TASK}/${SUBDIR}"

            if [[ ! -d "${MODEL_DIR}" ]]; then
                echo "   [skip] Checkpoint not found: ${MODEL_DIR}"
                SKIP=$((SKIP + 1))
                continue
            fi

            echo "  ── ${TASK} / ${METHOD}  (${MODEL_DIR})"
            rc=0
            python benchmark/tools/run_recovery_curve.py \
                --model_dir     "${MODEL_DIR}" \
                --task          "${TASK}" \
                --eval_steps    ${E3B_EVAL_STEPS} \
                --num_epochs    "${E3B_NUM_EPOCHS}" \
                --eval_backends "${E3B_EVAL_BACKENDS}" \
                --seq_len       "${SEQ_LEN}" \
                --batch_size    "${BATCH_SIZE}" \
                --dtype         "${DTYPE}" \
                --lr            "${LR}" \
                --out_csv       "${OUT_E3B}" \
            || rc=$?

            if   [[ $rc -eq 0 ]]; then OK=$((OK + 1))
            elif [[ $rc -eq 2 ]]; then SKIP=$((SKIP + 1)); echo "   [SKIP] ${TASK}/${METHOD}"
            else FAIL=$((FAIL + 1)); echo "   [FAILED] ${TASK}/${METHOD} (exit=${rc})"; fi
            echo ""
        done
    done

    echo "   E-3b out → ${OUT_E3B}"
    echo ""
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
echo "  expE done  success=${OK}  failed=${FAIL}  skipped=${SKIP}"
echo ""
echo "  Output:"
[[ "${PHASES}" == *"e2"*  ]] && echo "    E-2 alignment → ${OUT_EXPB}"
[[ "${PHASES}" == *"e3a"* ]] && echo "    E-3a training → ${OUT_E3A}"
[[ "${PHASES}" == *"e3b"* ]] && echo "    E-3b recovery → ${OUT_E3B}"
echo "══════════════════════════════════════════════════════════════════════"

[[ "${FAIL}" -eq 0 ]]
