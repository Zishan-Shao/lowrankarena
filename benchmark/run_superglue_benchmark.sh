#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_superglue_benchmark.sh
#
# Run all compression methods × multiple backends on SuperGLUE / HANS / ANLI and aggregate results.
# No fine-tuning; directly evaluates post-compression accuracy (zero-shot compression eval).
#
# Checkpoint logic:
#   Compression is done once (naive backend); checkpoint names carry the _naive suffix.
#   Backend loop loads from the naive checkpoint via --load_model_dir, only changing --backend.
#
# Task classification:
#   SuperGLUE-Core (included in average): boolq, rte_sg, wic, copa
#   Diagnostic (excluded from average):   cb   [high-variance, 56 examples, reference only]
#   Robustness:                           hans, anli_r1, anli_r2, anli_r3
#
# COPA notes:
#   Uses NLI two-choice scoring (non-standard classification).
#   Model: textattack/bert-base-uncased-MNLI (class 1 = entailment).
#   Calibration: uses copa's built-in train split (400 examples).
#
# CB notes:
#   Validation set has only 56 examples; high-variance diagnostic task; not included in SuperGLUE average.
#   Results are for reference only; interpret alongside multiple seeds.
#
# Compression methods (5 total):
#   dense, svd, fwsvd, drone, adasvd
#
# Usage:
#   cd lowrankarena/
#   bash benchmark/run_superglue_benchmark.sh
#
# Overridable variable examples:
#   TASKS="boolq hans copa"
#   METHODS="svd fwsvd"
#   BACKENDS="naive sdpa flashsvd"    # fp32 excludes flashsvd15 (would cast to fp16)
#   DTYPE=fp32 BACKENDS="naive sdpa flashsvd"   # fp32 precision test (without flashsvd15)
#   RECOMPRESS=true     # force recompression (ignore existing checkpoints)
#   OUT_CSV=experiments/expA.csv
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-experiments/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/superglue_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[log] → ${LOG_FILE}"

# ── Configuration ────────────────────────────────────────────────────────────
TASKS="${TASKS:-boolq rte_sg wic copa cb hans anli_r1 anli_r2 anli_r3}"
METHODS="${METHODS:-dense svd fwsvd drone adasvd}"
BACKENDS="${BACKENDS:-naive sdpa flashsvd flashsvd15}"

# Ranks for SVD / FWSVD / DRONE (identical to GLUE expA)
RANK_ATTN="${RANK_ATTN:-48}"
RANK_FFN="${RANK_FFN:-256}"
RANK_WO="${RANK_WO:-208}"
QKV_MODE="${QKV_MODE:-per_head}"

# AdaSVD budget (0.527 ≈ equivalent parameter count to the ranks above)
BUDGET="${BUDGET:-0.527}"
ADASVD_CALIB_SAMPLES="${ADASVD_CALIB_SAMPLES:-4000}"
ADASVD_STEPS="${ADASVD_STEPS:-800}"

CALIB_BATCHES="${CALIB_BATCHES:-16}"
DTYPE="${DTYPE:-bf16}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"

MODEL_BASE_DIR="${MODEL_BASE_DIR:-compressed_models/bert}"
OUT_CSV="${OUT_CSV:-experiments/expA.csv}"

# RECOMPRESS=true forces ignoring existing checkpoints and recompresses
RECOMPRESS="${RECOMPRESS:-false}"

# ── task → model_id mapping ───────────────────────────────────────────────────
_model_id_for_task() {
    case "$1" in
        boolq)   echo "howey/bert-base-uncased-boolq" ;;
        cb)      echo "textattack/bert-base-uncased-MNLI" ;;
        rte_sg)  echo "howey/bert-base-uncased-rte" ;;
        wic)     echo "rycecorn/Bert-fine-tuned-WiC" ;;
        copa)    echo "textattack/bert-base-uncased-MNLI" ;;
        hans)    echo "textattack/bert-base-uncased-MNLI" ;;
        anli_r1) echo "textattack/bert-base-uncased-MNLI" ;;
        anli_r2) echo "textattack/bert-base-uncased-MNLI" ;;
        anli_r3) echo "textattack/bert-base-uncased-MNLI" ;;
        *)       echo "" ;;
    esac
}

# Whether the task requires cross-task calibration (train_split=None)
_calib_task_for() {
    case "$1" in
        hans) echo "mnli" ;;   # hans has no train split
        *)    echo "" ;;
    esac
}

# Checkpoint subdirectory name — always carries _naive suffix (compression always saved with naive)
_ckpt_subdir() {
    local method="$1"
    case "${method}" in
        dense)  echo "dense_naive" ;;
        adasvd) echo "adasvd_b${BUDGET}_${QKV_MODE}_naive" ;;
        *)      echo "${method}_ra${RANK_ATTN}_rf${RANK_FFN}_rw${RANK_WO}_${QKV_MODE}_naive" ;;
    esac
}

echo "══════════════════════════════════════════════════════════════════════"
echo "  SuperGLUE / HANS / ANLI Compression Benchmark"
echo "  tasks:    ${TASKS}"
echo "  methods:  ${METHODS}"
echo "  backends: ${BACKENDS}"
echo "  ranks:    ra${RANK_ATTN}_rf${RANK_FFN}_rw${RANK_WO}  qkv=${QKV_MODE}"
echo "  adasvd:   budget=${BUDGET}  calib_samples=${ADASVD_CALIB_SAMPLES}  steps=${ADASVD_STEPS}"
echo "  dtype:    ${DTYPE}   seq_len: ${SEQ_LEN}   bs: ${BATCH_SIZE}"
echo "  models:   ${MODEL_BASE_DIR}  recompress: ${RECOMPRESS}"
echo "  out_csv:  ${OUT_CSV}"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

OK=0
FAIL=0

for TASK in ${TASKS}; do
    MODEL_ID="$(_model_id_for_task "${TASK}")"
    if [[ -z "${MODEL_ID}" ]]; then
        echo "[skip] Unknown task: ${TASK}"
        FAIL=$((FAIL + 1))
        continue
    fi

    CALIB_TASK="$(_calib_task_for "${TASK}")"
    EXTRA_ARGS=()
    [[ -n "${CALIB_TASK}" ]] && EXTRA_ARGS+=(--calib_task "${CALIB_TASK}")

    for METHOD in ${METHODS}; do
        SUBDIR="$(_ckpt_subdir "${METHOD}")"
        CKPT_DIR="${MODEL_BASE_DIR}/${TASK}/${SUBDIR}"

        # ── Step 1: ensure the naive checkpoint exists (not needed for dense) ────────
        if [[ "${METHOD}" != "dense" ]]; then
            if [[ "${RECOMPRESS}" == "true" || ! -d "${CKPT_DIR}" ]]; then
                echo "── ${TASK} / ${METHOD}  [compress → ${CKPT_DIR}]"

                if [[ "${METHOD}" == "adasvd" ]]; then
                    python src/encoders/compress.py \
                        --task                 "${TASK}" \
                        --model_id             "${MODEL_ID}" \
                        --method               adasvd \
                        --budget               "${BUDGET}" \
                        --qkv_mode             "${QKV_MODE}" \
                        --adasvd_calib_samples "${ADASVD_CALIB_SAMPLES}" \
                        --adasvd_steps         "${ADASVD_STEPS}" \
                        --backend              naive \
                        --dtype                "${DTYPE}" \
                        --seq_len              "${SEQ_LEN}" \
                        --batch_size           "${BATCH_SIZE}" \
                        --save_model \
                        --save_dir             "${MODEL_BASE_DIR}/${TASK}" \
                        --out_csv              "${OUT_CSV}" \
                        "${EXTRA_ARGS[@]}"
                else
                    python src/encoders/compress.py \
                        --task          "${TASK}" \
                        --model_id      "${MODEL_ID}" \
                        --method        "${METHOD}" \
                        --rank_attn     "${RANK_ATTN}" \
                        --rank_ffn      "${RANK_FFN}" \
                        --rank_wo       "${RANK_WO}" \
                        --qkv_mode      "${QKV_MODE}" \
                        --calib_batches "${CALIB_BATCHES}" \
                        --backend       naive \
                        --dtype         "${DTYPE}" \
                        --seq_len       "${SEQ_LEN}" \
                        --batch_size    "${BATCH_SIZE}" \
                        --save_model \
                        --save_dir      "${MODEL_BASE_DIR}/${TASK}" \
                        --out_csv       "${OUT_CSV}" \
                        "${EXTRA_ARGS[@]}"
                fi && OK=$((OK + 1)) || { FAIL=$((FAIL + 1)); continue; }
            else
                echo "── ${TASK} / ${METHOD}  [checkpoint exists: ${CKPT_DIR}]"
            fi
        fi

        # ── Step 2: iterate remaining backends (load from naive checkpoint) ──────────
        # sdpa is actually --backend naive --attn_mode sdpa, not an independent backend
        for BACKEND in ${BACKENDS}; do
            # naive was already evaluated and written to CSV during Step 1 compression; skip duplicate run
            [[ "${BACKEND}" == "naive" && "${METHOD}" != "dense" && -d "${CKPT_DIR}" && "${RECOMPRESS}" != "true" ]] && \
                echo "   [skip naive — already in CSV from compress step]" && continue

            # Translate sdpa → --backend naive --attn_mode sdpa
            if [[ "${BACKEND}" == "sdpa" ]]; then
                BACKEND_ARGS=(--backend naive --attn_mode sdpa)
            else
                BACKEND_ARGS=(--backend "${BACKEND}")
            fi

            echo "   ── ${TASK} / ${METHOD} / ${BACKEND}"

            if [[ "${METHOD}" == "dense" ]]; then
                python src/encoders/compress.py \
                    --task       "${TASK}" \
                    --model_id   "${MODEL_ID}" \
                    --method     dense \
                    --dtype      "${DTYPE}" \
                    --seq_len    "${SEQ_LEN}" \
                    --batch_size "${BATCH_SIZE}" \
                    --out_csv    "${OUT_CSV}" \
                    "${BACKEND_ARGS[@]}" \
                    "${EXTRA_ARGS[@]}" \
                    && OK=$((OK + 1)) || FAIL=$((FAIL + 1))
            else
                python src/encoders/compress.py \
                    --task           "${TASK}" \
                    --model_id       "${MODEL_ID}" \
                    --method         "${METHOD}" \
                    --load_model_dir "${CKPT_DIR}" \
                    --dtype          "${DTYPE}" \
                    --seq_len        "${SEQ_LEN}" \
                    --batch_size     "${BATCH_SIZE}" \
                    --out_csv        "${OUT_CSV}" \
                    "${BACKEND_ARGS[@]}" \
                    "${EXTRA_ARGS[@]}" \
                    && OK=$((OK + 1)) || FAIL=$((FAIL + 1))
            fi
        done
        echo ""
    done
done

echo "══════════════════════════════════════════════════════════════════════"
echo "  Done  success=${OK}  failed=${FAIL}"
echo "  Results → ${OUT_CSV}"
echo "══════════════════════════════════════════════════════════════════════"

[[ "${FAIL}" -eq 0 ]]
