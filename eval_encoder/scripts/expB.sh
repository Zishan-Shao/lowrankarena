#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# expB.sh  —  Backend Performance Microbenchmark
#
# 固定模型、固定输入形状，遍历 4 个 backend，量化：
#   latency / throughput / peak_mem / FLOPs / MFU / arithmetic_intensity
#
# 设计原则
# ────────────────────────────────────────────────────────────────────────────
#   • 不重新压缩：从已有 naive checkpoint load（压缩统一由 expA.sh 完成）
#   • 无 checkpoint = skip（打印警告，不 abort）
#   • 4 个 backend：naive(einsum) / naive(sdpa) / flashsvd / flashsvd15
#     → analyze_compute.py 接受 backend=sdpa，内部自动翻译为 --attn_mode sdpa
#   • 单一输出：eval_encoder/eval_results/expB_backend.csv
#     （同时含 latency/memory 列 和 FLOPs/MFU 列，所有图从同一文件读取）
#
# 取代
#   run_expA_sdpa.sh   （旧：只跑 analyze_compute.py，输出已归档至 _retired/）
#   run_sdpa_ablation.sh （旧：只跑 run_encoder_benchmark.py, 输出 encoder_runs.csv）
#
# 用法
# ────────────────────────────────────────────────────────────────────────────
#   cd lowrankarena/
#   bash eval_encoder/scripts/expB.sh
#
# 可覆盖变量
#   TASKS="mnli stsb"           # 任务子集（默认 8 个 GLUE 任务）
#   METHODS="svd adasvd"        # 方法子集
#   BACKENDS="naive flashsvd15" # backend 子集
#   DTYPE=fp32                  # 精度（flashsvd15 只支持 bf16/fp16，fp32 会跳过）
#   SEQ_LEN=256
#   BATCH_SIZE=16
#   WARMUP=10
#   MEASURE=50
#   MODEL_BASE_DIR=eval_encoder/models
#   OUT_CSV=eval_encoder/eval_results/expB_backend.csv
#
# checkpoint 命名规则（与 expA / glue_pipeline.py 一致）
#   SVD/FWSVD/DRONE : {task}/{method}_ra{ra}_rf{rf}_rw{rw}_{qkv_mode}_naive
#   AdaSVD          : {task}/adasvd_b{budget}_{qkv_mode}_naive
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ── 配置 ──────────────────────────────────────────────────────────────────────
TASKS="${TASKS:-cola sst2 mrpc qqp mnli qnli rte stsb}"
METHODS="${METHODS:-svd fwsvd drone adasvd}"
BACKENDS="${BACKENDS:-naive sdpa flashsvd flashsvd15}"
DTYPE="${DTYPE:-bf16}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WARMUP="${WARMUP:-5}"
MEASURE="${MEASURE:-16}"   # 16 × bs=32 = 512 calib samples

# 与 expA / glue_pipeline.py 命名一致
RANK_ATTN="${RANK_ATTN:-48}"
RANK_FFN="${RANK_FFN:-256}"
RANK_WO="${RANK_WO:-208}"
QKV_MODE="${QKV_MODE:-per_head}"
BUDGET="${BUDGET:-0.527}"

MODEL_BASE_DIR="${MODEL_BASE_DIR:-eval_encoder/models}"
OUT_CSV="${OUT_CSV:-eval_encoder/eval_results/expB_backend.csv}"

# ── checkpoint 子目录名 ────────────────────────────────────────────────────────
_model_subdir() {
    local method="$1"
    if [[ "${method}" == "adasvd" ]]; then
        echo "adasvd_b${BUDGET}_${QKV_MODE}_naive"
    else
        echo "${method}_ra${RANK_ATTN}_rf${RANK_FFN}_rw${RANK_WO}_${QKV_MODE}_naive"
    fi
}

# flashsvd15 仅支持 bf16 / fp16；fp32 时跳过
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
echo "  tasks:    ${TASKS}"
echo "  methods:  ${METHODS}"
echo "  backends: ${BACKENDS}"
echo "  dtype:    ${DTYPE}   seq_len: ${SEQ_LEN}   bs: ${BATCH_SIZE}"
echo "  config:   ra${RANK_ATTN}_rf${RANK_FFN}_rw${RANK_WO}  qkv=${QKV_MODE}  budget=${BUDGET}"
echo "  models:   ${MODEL_BASE_DIR}"
echo "  out_csv:  ${OUT_CSV}"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

OK=0
FAIL=0
SKIP=0

for TASK in ${TASKS}; do
    for METHOD in ${METHODS}; do
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

            echo "   → backend=${BACKEND}"

            python eval_encoder/scripts/analyze_compute.py \
                --model_dir  "${MODEL_DIR}" \
                --task       "${TASK}" \
                --backend    "${BACKEND}" \
                --dtype      "${DTYPE}" \
                --seq_len    "${SEQ_LEN}" \
                --batch_size "${BATCH_SIZE}" \
                --warmup     "${WARMUP}" \
                --measure    "${MEASURE}" \
                --out_csv    "${OUT_CSV}" \
            && OK=$((OK + 1)) || { FAIL=$((FAIL + 1)); echo "   [FAILED] ${TASK}/${METHOD}/${BACKEND}"; }
        done
        echo ""
    done
done

echo "══════════════════════════════════════════════════════════════════════"
echo "  完成  成功=${OK}  失败=${FAIL}  跳过=${SKIP}"
echo "  结果 → ${OUT_CSV}"
echo "══════════════════════════════════════════════════════════════════════"

[[ "${FAIL}" -eq 0 ]]
