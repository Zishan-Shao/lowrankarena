#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# expC.sh  —  Scaling Experiment
#
# Fixed checkpoint, measuring throughput / latency / peak_mem trends as
# seq_len / batch_size vary (4-backend comparison).
#
# Design principles
# ────────────────────────────────────────────────────────────────────────────
#   • Reuses expA artifacts (no recompression); missing checkpoints cause [skip], not abort
#   • Only 2 representative methods: svd (uniform rank) + adasvd (heterogeneous rank)
#   • Only 2 representative tasks: mnli (long sentences) + stsb (short sentences, different padding characteristics)
#   • Calls analyze_compute.py directly (same as expB, no new measurement protocol)
#   • Two independent phases:
#       seqlen — batch=32 fixed, sweep seq_len = 128 256 384 512
#       batch  — seq=512 fixed, sweep batch_size = 8 16 32 64
#   • input_mode=synthetic (0% padding, consistent with expB, fair backend comparison)
#   • repeat=3 (mean±std variance control, consistent with expB)
#   • Clears corresponding CSV before each run (avoids schema-incompatible stale rows)
#
# Note: dense has been removed from default METHODS — dense model path naming differs
#    from SVD methods and requires extra handling. Use run_encoder_benchmark.py for dense baseline.
#
# Note: seq_len=768 is not yet supported — BERT-base max_position_embeddings=512
#    and exceeding it causes an error. ModernBERT supports longer sequences
#    but expC currently only runs BERT-base.
#
# Usage
# ────────────────────────────────────────────────────────────────────────────
#   cd lowrankarena/
#   bash benchmark/expC.sh                  # full run (seqlen + batch)
#   PHASES=seqlen bash benchmark/expC.sh    # seq_len sweep only
#   PHASES=batch  bash benchmark/expC.sh    # batch sweep only
#
# Overridable variables
#   PHASES="seqlen batch"
#   TASKS="mnli stsb"
#   METHODS="svd adasvd"            # dense removed from default (different path naming)
#   BACKENDS="naive sdpa flashsvd flashsvd15"
#   SEQ_LENS="128 256 384 512"      # scan points for seqlen phase
#   BATCH_SIZES="8 16 32 64"        # scan points for batch phase
#   BATCH_FIXED=32                  # fixed batch for seqlen phase
#   SEQ_FIXED=512                   # fixed seq_len for batch phase
#   DTYPE=bf16
#   INPUT_MODE=synthetic            # real | synthetic (default synthetic, 0% padding)
#   REPEAT=3                        # number of repeated measurements (mean±std, consistent with expB)
#   WARMUP=10  MEASURE=50           # consistent with expB measurement protocol
#   ALIGN=0                         # 0=off (default), 1=on (adds --check_alignment, records logit_max_diff)
#   MODEL_BASE_DIR=compressed_models/bert
#   OUT_SEQLEN=experiments/expC_seqlen.csv
#   OUT_BATCH=experiments/expC_batch.csv
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-experiments/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/expC_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[log] → ${LOG_FILE}"

# ── Configuration ─────────────────────────────────────────────────────────────
PHASES="${PHASES:-seqlen batch}"
TASKS="${TASKS:-mnli stsb}"
# dense removed: its model path naming differs from SVD checkpoints and causes path lookup failure
METHODS="${METHODS:-svd adasvd}"
BACKENDS="${BACKENDS:-naive sdpa flashsvd flashsvd15}"
DTYPE="${DTYPE:-bf16}"
WARMUP="${WARMUP:-10}"
MEASURE="${MEASURE:-50}"
# synthetic: 0% padding, eliminates padding-rate interference on backend throughput (consistent with expB)
INPUT_MODE="${INPUT_MODE:-synthetic}"
# Number of independent repetitions: mean±std (consistent with expB measurement protocol)
REPEAT="${REPEAT:-3}"

# seqlen phase parameters
SEQ_LENS="${SEQ_LENS:-128 256 384 512}"
BATCH_FIXED="${BATCH_FIXED:-32}"

# batch phase parameters
BATCH_SIZES="${BATCH_SIZES:-8 16 32 64}"
SEQ_FIXED="${SEQ_FIXED:-512}"

# Checkpoint naming (consistent with expA / glue_pipeline.py)
RANK_ATTN="${RANK_ATTN:-48}"
RANK_FFN="${RANK_FFN:-256}"
RANK_WO="${RANK_WO:-208}"
QKV_MODE="${QKV_MODE:-per_head}"
BUDGET="${BUDGET:-0.527}"

MODEL_BASE_DIR="${MODEL_BASE_DIR:-compressed_models/bert}"
OUT_SEQLEN="${OUT_SEQLEN:-experiments/expC_seqlen.csv}"
OUT_BATCH="${OUT_BATCH:-experiments/expC_batch.csv}"
# Alignment check: 0=off (default); 1=on (each job loads naive once more for logit diff)
# Usage: ALIGN=1 bash expC.sh
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

# flashsvd15 only supports bf16 / fp16
_skip_backend() {
    local backend="$1"
    if [[ "${backend}" == "flashsvd15" && "${DTYPE}" == "fp32" ]]; then
        echo "true"
    else
        echo "false"
    fi
}

# ── Internal: run analyze_compute.py for one (method, task, backend) combination ────
# Return codes: 0=success  2=skip (checkpoint missing / backend unsupported)  other=failure
_run_one() {
    local method="$1" task="$2" backend="$3" seq_len="$4" batch_size="$5" out_csv="$6"

    local subdir model_dir
    subdir="$(_model_subdir "${method}")"
    model_dir="${MODEL_BASE_DIR}/${task}/${subdir}"

    if [[ ! -d "${model_dir}" ]]; then
        echo "      [skip] Checkpoint not found: ${model_dir}"
        return 2
    fi

    if [[ "$(_skip_backend "${backend}")" == "true" ]]; then
        echo "      [skip] ${backend} requires bf16/fp16, current dtype=${DTYPE}"
        return 2
    fi

    python benchmark/analysis/analyze_compute.py \
        --model_dir  "${model_dir}" \
        --task       "${task}" \
        --backend    "${backend}" \
        --input_mode "${INPUT_MODE}" \
        --repeat     "${REPEAT}" \
        --dtype      "${DTYPE}" \
        --seq_len    "${seq_len}" \
        --batch_size "${batch_size}" \
        --warmup     "${WARMUP}" \
        --measure    "${MEASURE}" \
        --out_csv    "${out_csv}" \
        ${ALIGN_FLAG}
}

# ── Banner ────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
echo "  expC — Scaling Experiment"
echo "  phases:     ${PHASES}"
echo "  tasks:      ${TASKS}   methods: ${METHODS}"
echo "  backends:   ${BACKENDS}   dtype: ${DTYPE}"
echo "  input_mode: ${INPUT_MODE}   repeat: ${REPEAT}  (mean±std)"
echo "  warmup:     ${WARMUP}   measure: ${MEASURE}"
echo "  align:      ${ALIGN}  (ALIGN=1 → --check_alignment, writes logit_max_diff to CSV)"
echo "  config:     ra${RANK_ATTN}_rf${RANK_FFN}_rw${RANK_WO}  qkv=${QKV_MODE}  budget=${BUDGET}"
echo "  models:     ${MODEL_BASE_DIR}"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

OK=0
FAIL=0
SKIP=0

# ── Phase A: seq_len sweep ─────────────────────────────────────────────────────
if [[ "${PHASES}" == *"seqlen"* ]]; then
    echo "══ Phase A: seq_len sweep (batch=${BATCH_FIXED}) ══════════════════"
    echo "   seq_lens: ${SEQ_LENS}"
    echo "   out_csv:  ${OUT_SEQLEN}"
    echo ""

    # Clear old CSV to avoid schema-incompatible historical rows
    mkdir -p "$(dirname "${OUT_SEQLEN}")"
    rm -f "${OUT_SEQLEN}"

    for SEQ_LEN in ${SEQ_LENS}; do
        for TASK in ${TASKS}; do
            echo "  seq_len=${SEQ_LEN}  task=${TASK}"
            for METHOD in ${METHODS}; do
                for BACKEND in ${BACKENDS}; do
                    echo "    → ${METHOD} / ${BACKEND}"
                    rc=0; _run_one "${METHOD}" "${TASK}" "${BACKEND}" \
                                   "${SEQ_LEN}" "${BATCH_FIXED}" "${OUT_SEQLEN}" || rc=$?
                    if   [[ $rc -eq 0 ]]; then OK=$((OK + 1))
                    elif [[ $rc -eq 2 ]]; then SKIP=$((SKIP + 1))
                    else FAIL=$((FAIL + 1)); echo "    [FAILED] ${METHOD}/${TASK}/${BACKEND}/seq${SEQ_LEN}"; fi
                done
            done
        done
        echo ""
    done
fi

# ── Phase B: batch_size sweep ──────────────────────────────────────────────────
if [[ "${PHASES}" == *"batch"* ]]; then
    echo "══ Phase B: batch_size sweep (seq_len=${SEQ_FIXED}) ═══════════════"
    echo "   batch_sizes: ${BATCH_SIZES}"
    echo "   out_csv:     ${OUT_BATCH}"
    echo ""

    # Clear old CSV to avoid schema-incompatible historical rows
    mkdir -p "$(dirname "${OUT_BATCH}")"
    rm -f "${OUT_BATCH}"

    for BATCH_SIZE in ${BATCH_SIZES}; do
        for TASK in ${TASKS}; do
            echo "  batch=${BATCH_SIZE}  task=${TASK}"
            for METHOD in ${METHODS}; do
                for BACKEND in ${BACKENDS}; do
                    echo "    → ${METHOD} / ${BACKEND}"
                    rc=0; _run_one "${METHOD}" "${TASK}" "${BACKEND}" \
                                   "${SEQ_FIXED}" "${BATCH_SIZE}" "${OUT_BATCH}" || rc=$?
                    if   [[ $rc -eq 0 ]]; then OK=$((OK + 1))
                    elif [[ $rc -eq 2 ]]; then SKIP=$((SKIP + 1))
                    else FAIL=$((FAIL + 1)); echo "    [FAILED] ${METHOD}/${TASK}/${BACKEND}/bs${BATCH_SIZE}"; fi
                done
            done
        done
        echo ""
    done
fi

echo "══════════════════════════════════════════════════════════════════════"
echo "  expC done  success=${OK}  failed=${FAIL}  skipped=${SKIP}"
echo ""
echo "  Output:"
[[ "${PHASES}" == *"seqlen"* ]] && echo "    seq_len sweep → ${OUT_SEQLEN}"
[[ "${PHASES}" == *"batch"*  ]] && echo "    batch sweep   → ${OUT_BATCH}"
echo "══════════════════════════════════════════════════════════════════════"

[[ "${FAIL}" -eq 0 ]]
