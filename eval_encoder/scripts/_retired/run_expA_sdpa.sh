#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_expA_sdpa.sh
#
# 补充 SDPA 消融行到 expA_backend.csv。
# 从已有 checkpoint load，不重新压缩。
#
# 目的：拆分 FlashSVD 加速来源：
#   naive(einsum) → naive(SDPA)  : Flash Attention 本身的收益
#   naive(SDPA)   → flashsvd15   : Triton 低秩投影融合的额外收益
#
# 模型路径规律：
#   SVD/FWSVD/DRONE : eval_encoder/models/{task}/svd_ra48_rf256_rw208_per_head_naive
#   AdaSVD          : eval_encoder/models/{task}/adasvd_b0.527_per_head_naive
#
# 用法：
#   cd lowrankarena/
#   bash eval_encoder/scripts/run_expA_sdpa.sh
#
# 可覆盖变量：
#   TASKS="mnli stsb"
#   METHODS="svd adasvd"
#   MODEL_BASE_DIR=eval_encoder/models
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ── 配置 ─────────────────────────────────────────────────────────────────────
TASKS="${TASKS:-mnli stsb}"
METHODS="${METHODS:-svd adasvd}"
BACKENDS="${BACKENDS:-naive sdpa flashsvd flashsvd15}"
DTYPE="${DTYPE:-bf16}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WARMUP="${WARMUP:-10}"
MEASURE="${MEASURE:-50}"

# 与 glue_pipeline.py / run_sdpa_ablation.sh 命名一致
RANK_ATTN="${RANK_ATTN:-48}"
RANK_FFN="${RANK_FFN:-256}"
RANK_WO="${RANK_WO:-208}"
QKV_MODE="${QKV_MODE:-per_head}"
BUDGET="${BUDGET:-0.527}"

MODEL_BASE_DIR="${MODEL_BASE_DIR:-eval_encoder/models}"
OUT_CSV="${OUT_CSV:-eval_encoder/eval_results/expA_backend.csv}"

# ── 模型子目录名（与 glue_pipeline 命名一致）─────────────────────────────────
_model_subdir() {
    local method="$1"
    if [[ "${method}" == "adasvd" ]]; then
        echo "adasvd_b${BUDGET}_${QKV_MODE}_naive"
    else
        echo "${method}_ra${RANK_ATTN}_rf${RANK_FFN}_rw${RANK_WO}_${QKV_MODE}_naive"
    fi
}

echo "══════════════════════════════════════════════════════════════"
echo "  expA backend 实验（load from checkpoint）"
echo "  tasks=${TASKS}  methods=${METHODS}  backends=${BACKENDS}"
echo "  dtype=${DTYPE}  seq_len=${SEQ_LEN}  bs=${BATCH_SIZE}"
echo "  out_csv=${OUT_CSV}"
echo "══════════════════════════════════════════════════════════════"

for TASK in ${TASKS}; do
    for METHOD in ${METHODS}; do
        MODEL_SUBDIR="$(_model_subdir "${METHOD}")"
        MODEL_DIR="${MODEL_BASE_DIR}/${TASK}/${MODEL_SUBDIR}"

        if [[ ! -d "${MODEL_DIR}" ]]; then
            echo "[skip] Model not found: ${MODEL_DIR}"
            continue
        fi

        for BACKEND in ${BACKENDS}; do
            echo ""
            echo "── ${METHOD} / ${TASK} / ${BACKEND}  (load: ${MODEL_DIR})"

            python eval_encoder/scripts/analyze_compute.py \
                --model_dir  "${MODEL_DIR}" \
                --task       "${TASK}" \
                --backend    "${BACKEND}" \
                --dtype      "${DTYPE}" \
                --seq_len    "${SEQ_LEN}" \
                --batch_size "${BATCH_SIZE}" \
                --warmup     "${WARMUP}" \
                --measure    "${MEASURE}" \
                --out_csv    "${OUT_CSV}"
        done
    done
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  完成 → ${OUT_CSV}"
echo "══════════════════════════════════════════════════════════════"
