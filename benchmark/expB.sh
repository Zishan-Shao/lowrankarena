#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# expB.sh  —  Backend Performance Microbenchmark
#
# Fixed model and input shape, sweep 4 backends, measuring:
#   latency / throughput / peak_mem / FLOPs / MFU / arithmetic_intensity
#
# Design principles (benchmark fairness)
# ────────────────────────────────────────────────────────────────────────────
#   Main table: fixed SVD checkpoint, backend comparison only
#     • METHODS defaults to "svd" (single rank config), eliminates rank-allocation interference
#     • Backend comparison is apples-to-apples: same checkpoint, different inference paths
#     • For rank-strategy sensitivity analysis, override METHODS="svd fwsvd drone adasvd"
#
#   Input distribution: dual-mode reporting
#     • INPUT_MODES="real synthetic" (default: run both)
#       - real:      real task validation set (natural seq padding, 86-98% padding rate)
#       - synthetic: random tokens + all-1 mask (0% padding, fully-utilized input)
#       Both together answer: "which backend is stronger on real vs fully-loaded inputs"
#
#   Repeated measurement: variance control
#     • REPEAT=3 (default, 3 independent measurements each with full warmup)
#     • CSV outputs latency_ms_std / throughput_sps_std
#     • Main table reports mean, appendix reports std — sufficient to answer "are results stable"
#
#   Other
#     • No recompression: loads from existing naive checkpoint (compression done by expA.sh)
#     • No checkpoint = skip (prints warning, does not abort)
#     • SDPA path is already recorded by analyze_compute.py in the CSV notes field
#
# Usage
# ────────────────────────────────────────────────────────────────────────────
#   cd lowrankarena/
#   bash benchmark/expB.sh                              # default (svd × 4 backends × real+synthetic × 3 repeats)
#   METHODS="dense svd fwsvd drone adasvd" bash benchmark/expB.sh  # full sweep incl. dense
#   INPUT_MODES=real REPEAT=1 bash benchmark/expB.sh   # quick single real-data run
#   TASKS="mnli stsb" bash benchmark/expB.sh           # task subset
#
# Overridable variables
#   TASKS="mnli stsb"               # task subset (default: 8 GLUE tasks)
#   METHODS="dense svd"             # method (default: dense baseline + svd)
#   BACKENDS="naive sdpa flashsvd flashsvd15"
#   INPUT_MODES="real synthetic"    # input distribution (default: run both)
#   REPEAT=3                        # number of repeated measurements (default 3)
#   DTYPE=fp32                      # dtype (flashsvd15 only supports bf16/fp16; fp32 will be skipped)
#   SEQ_LEN=512
#   BATCH_SIZE=32
#   WARMUP=10
#   MEASURE=50
#   MODEL_BASE_DIR=compressed_models/bert
#   OUT_CSV=experiments/results/expB.csv
#   ALIGN=0                         # 0=off (default), 1=on (adds --check_alignment, records logit_max_diff)
#
# Checkpoint naming convention (consistent with expA / glue_pipeline.py)
#   SVD/FWSVD/DRONE : {task}/{method}_ra{ra}_rf{rf}_rw{rw}_{qkv_mode}_naive
#   AdaSVD          : {task}/adasvd_b{budget}_{qkv_mode}_naive
#
# Note: new fields input_mode / n_repeats / latency_ms_std / throughput_sps_std added
#   If an old expB.csv already exists, delete it and rerun (schema incompatible with new header)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-experiments/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/expB_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[log] → ${LOG_FILE}"

# ── Configuration ─────────────────────────────────────────────────────────────
TASKS="${TASKS:-cola sst2 mrpc qqp mnli qnli rte stsb}"
# Main table: dense baseline + fixed svd checkpoint, backend comparison only
# dense: loads HuggingFace model directly (no checkpoint needed)
# For rank-strategy sensitivity, override METHODS="dense svd fwsvd drone adasvd"
METHODS="${METHODS:-dense svd}"
BACKENDS="${BACKENDS:-naive sdpa flashsvd flashsvd15}"
# Dual mode: real (real data, natural padding) + synthetic (all valid tokens, 0% padding)
INPUT_MODES="${INPUT_MODES:-real synthetic}"
# Number of repeated measurements (independent warmup+measure, reports mean±std)
REPEAT="${REPEAT:-3}"
DTYPE="${DTYPE:-bf16}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WARMUP="${WARMUP:-10}"
MEASURE="${MEASURE:-50}"

# Consistent with expA / glue_pipeline.py naming
RANK_ATTN="${RANK_ATTN:-48}"
RANK_FFN="${RANK_FFN:-256}"
RANK_WO="${RANK_WO:-208}"
QKV_MODE="${QKV_MODE:-per_head}"
BUDGET="${BUDGET:-0.527}"

MODEL_BASE_DIR="${MODEL_BASE_DIR:-compressed_models/bert}"
OUT_CSV="${OUT_CSV:-experiments/results/expB.csv}"
# Alignment check: 0=off (default, no speed impact); 1=on (each job loads naive once more for logit diff)
# Usage: ALIGN=1 bash expB.sh
ALIGN="${ALIGN:-0}"
ALIGN_FLAG=""
[[ "${ALIGN}" -eq 1 ]] && ALIGN_FLAG="--check_alignment"

# ── Checkpoint subdirectory name ──────────────────────────────────────────────
_model_subdir() {
    local method="$1"
    if [[ "${method}" == "adasvd" ]]; then
        echo "adasvd_b${BUDGET}_${QKV_MODE}_naive"
    else
        echo "${method}_ra${RANK_ATTN}_rf${RANK_FFN}_rw${RANK_WO}_${QKV_MODE}_naive"
    fi
}

# flashsvd15 only supports bf16 / fp16; skip when fp32
_skip_backend() {
    local backend="$1"
    if [[ "${backend}" == "flashsvd15" && "${DTYPE}" == "fp32" ]]; then
        echo "true"
    else
        echo "false"
    fi
}

# ── Banner ────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
echo "  expB — Backend Performance Microbenchmark"
echo "  tasks:       ${TASKS}"
echo "  methods:     ${METHODS}"
echo "  backends:    ${BACKENDS}"
echo "  input_modes: ${INPUT_MODES}"
echo "  repeat:      ${REPEAT}  (mean±std across independent runs)"
echo "  dtype:       ${DTYPE}   seq_len: ${SEQ_LEN}   bs: ${BATCH_SIZE}"
echo "  warmup:      ${WARMUP}  measure: ${MEASURE}"
echo "  config:      ra${RANK_ATTN}_rf${RANK_FFN}_rw${RANK_WO}  qkv=${QKV_MODE}  budget=${BUDGET}"
echo "  models:      ${MODEL_BASE_DIR}"
echo "  out_csv:     ${OUT_CSV}"
echo "  align:       ${ALIGN}  (ALIGN=1 → --check_alignment, writes logit_max_diff to CSV)"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

OK=0
FAIL=0
SKIP=0

for TASK in ${TASKS}; do
    for METHOD in ${METHODS}; do

        # ── Dense baseline: prefer local checkpoint, fall back to HuggingFace ──
        if [[ "${METHOD}" == "dense" ]]; then
            # Priority: task-specific > global > HuggingFace
            _DENSE_MODEL_ARG=""
            for _candidate in \
                "${MODEL_BASE_DIR}/dense/${TASK}/dense_rNone_naive" \
                "${MODEL_BASE_DIR}/dense/${TASK}/dense_naive" \
                "${MODEL_BASE_DIR}/dense/dense_rNone_naive" \
                "${MODEL_BASE_DIR}/dense/dense_naive"
            do
                if [[ -d "${_candidate}" ]]; then
                    _DENSE_MODEL_ARG="--model_dir ${_candidate}"
                    break
                fi
            done
            if [[ -n "${_DENSE_MODEL_ARG}" ]]; then
                echo "── ${TASK} / dense  (local: ${_candidate})"
            else
                echo "── ${TASK} / dense  (HuggingFace fallback)"
            fi
            for BACKEND in ${BACKENDS}; do
                if [[ "$(_skip_backend "${BACKEND}")" == "true" ]]; then
                    echo "   [skip] ${BACKEND} requires bf16/fp16, current dtype=${DTYPE}"
                    SKIP=$((SKIP + 1))
                    continue
                fi
                for INPUT_MODE in ${INPUT_MODES}; do
                    echo "   → backend=${BACKEND}  input_mode=${INPUT_MODE}"
                    python benchmark/analysis/analyze_compute.py \
                        --method     dense \
                        ${_DENSE_MODEL_ARG} \
                        --task       "${TASK}" \
                        --backend    "${BACKEND}" \
                        --input_mode "${INPUT_MODE}" \
                        --repeat     "${REPEAT}" \
                        --dtype      "${DTYPE}" \
                        --seq_len    "${SEQ_LEN}" \
                        --batch_size "${BATCH_SIZE}" \
                        --warmup     "${WARMUP}" \
                        --measure    "${MEASURE}" \
                        --out_csv    "${OUT_CSV}" \
                    && OK=$((OK + 1)) || { FAIL=$((FAIL + 1)); echo "   [FAILED] ${TASK}/dense/${BACKEND}/${INPUT_MODE}"; }
                done
            done
            echo ""
            continue
        fi

        # ── Compressed methods: require checkpoint ──────────────────────────
        SUBDIR="$(_model_subdir "${METHOD}")"
        MODEL_DIR="${MODEL_BASE_DIR}/${TASK}/${SUBDIR}"

        if [[ ! -d "${MODEL_DIR}" ]]; then
            echo "[skip] Checkpoint not found: ${MODEL_DIR}"
            SKIP=$((SKIP + 1))
            continue
        fi

        echo "── ${TASK} / ${METHOD}  (${MODEL_DIR})"

        for BACKEND in ${BACKENDS}; do
            if [[ "$(_skip_backend "${BACKEND}")" == "true" ]]; then
                echo "   [skip] ${BACKEND} requires bf16/fp16, current dtype=${DTYPE}"
                SKIP=$((SKIP + 1))
                continue
            fi

            for INPUT_MODE in ${INPUT_MODES}; do
                echo "   → backend=${BACKEND}  input_mode=${INPUT_MODE}"

                python benchmark/analysis/analyze_compute.py \
                    --model_dir  "${MODEL_DIR}" \
                    --task       "${TASK}" \
                    --backend    "${BACKEND}" \
                    --input_mode "${INPUT_MODE}" \
                    --repeat     "${REPEAT}" \
                    --dtype      "${DTYPE}" \
                    --seq_len    "${SEQ_LEN}" \
                    --batch_size "${BATCH_SIZE}" \
                    --warmup     "${WARMUP}" \
                    --measure    "${MEASURE}" \
                    --out_csv    "${OUT_CSV}" \
                    ${ALIGN_FLAG} \
                && OK=$((OK + 1)) || { FAIL=$((FAIL + 1)); echo "   [FAILED] ${TASK}/${METHOD}/${BACKEND}/${INPUT_MODE}"; }
            done
        done
        echo ""
    done
done

echo "══════════════════════════════════════════════════════════════════════"
echo "  Done  success=${OK}  failed=${FAIL}  skipped=${SKIP}"
echo "  Results → ${OUT_CSV}"
echo "══════════════════════════════════════════════════════════════════════"

# ── Auto-extract E-1/E-2/E-3 sub-CSVs (only when main CSV exists) ────────────
OUT_DIR="$(dirname "${OUT_CSV}")"
if [[ -f "${OUT_CSV}" ]]; then
    echo ""
    echo "── Extract E-1/E-2/E-3 sub-CSVs → ${OUT_DIR}"
    python benchmark/analysis/collect_expBE.py \
        --input  "${OUT_CSV}" \
        --outdir "${OUT_DIR}" \
    && echo "   expE1_alignment.csv / expE2_padding.csv / expE3_repeatability.csv" \
    || echo "   [warn] collect_expBE.py failed, please run manually"
    echo ""
fi

[[ "${FAIL}" -eq 0 ]]
