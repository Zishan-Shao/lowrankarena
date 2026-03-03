#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# expB.sh  —  Backend Performance Microbenchmark
#
# 固定模型、固定输入形状，遍历 4 个 backend，量化：
#   latency / throughput / peak_mem / FLOPs / MFU / arithmetic_intensity
#
# 设计原则（benchmark 公平性）
# ────────────────────────────────────────────────────────────────────────────
#   主表设计：固定 SVD checkpoint，只比 backend
#     • METHODS 默认 "svd"（单一 rank 配置），排除 method 对 rank 分配的干扰
#     • backend 比较是 apples-to-apples：相同 checkpoint，不同推理路径
#     • 如需 rank-strategy sensitivity 分析，覆盖 METHODS="svd fwsvd drone adasvd"
#
#   输入分布：双模式报告
#     • INPUT_MODES="real synthetic"（默认同时跑两个）
#       - real:      真实任务 validation set（自然 seq padding，86-98% padding rate）
#       - synthetic: 随机 token + all-1 mask（0% padding，fully-utilized input）
#       两者一起才能回答："真实数据 vs 满载输入下，哪个 backend 更强"
#
#   重复测量：variance control
#     • REPEAT=3（默认，3 轮独立测量，各含完整 warmup）
#     • CSV 输出 latency_ms_std / throughput_sps_std
#     • 主表报 mean，附录报 std — 足以回答 reviewer "结果是否稳定"
#
#   其他
#     • 不重新压缩：从已有 naive checkpoint load（压缩统一由 expA.sh 完成）
#     • 无 checkpoint = skip（打印警告，不 abort）
#     • SDPA path 已由 analyze_compute.py 记录在 CSV notes 字段
#
# 用法
# ────────────────────────────────────────────────────────────────────────────
#   cd lowrankarena/
#   bash eval_encoder/scripts/expB.sh                              # 默认（svd × 4 backend × real+synthetic × 3 repeats）
#   METHODS="svd fwsvd drone adasvd" bash eval_encoder/scripts/expB.sh  # rank-strategy sensitivity
#   INPUT_MODES=real REPEAT=1 bash eval_encoder/scripts/expB.sh   # 快速单次 real-data run
#   TASKS="mnli stsb" bash eval_encoder/scripts/expB.sh           # 任务子集
#
# 可覆盖变量
#   TASKS="mnli stsb"               # 任务子集（默认 8 个 GLUE 任务）
#   METHODS="svd"                   # 方法（默认 "svd"，主表只比 backend）
#   BACKENDS="naive sdpa flashsvd flashsvd15"
#   INPUT_MODES="real synthetic"    # 输入分布（默认两个都跑）
#   REPEAT=3                        # 重复测量次数（默认 3）
#   DTYPE=fp32                      # 精度（flashsvd15 只支持 bf16/fp16，fp32 会跳过）
#   SEQ_LEN=512
#   BATCH_SIZE=32
#   WARMUP=10
#   MEASURE=50
#   MODEL_BASE_DIR=eval_encoder/models
#   OUT_CSV=eval_encoder/eval_results/expB.csv
#   ALIGN=0                         # 0=关（默认），1=开（加 --check_alignment，记录 logit_max_diff）
#
# checkpoint 命名规则（与 expA / glue_pipeline.py 一致）
#   SVD/FWSVD/DRONE : {task}/{method}_ra{ra}_rf{rf}_rw{rw}_{qkv_mode}_naive
#   AdaSVD          : {task}/adasvd_b{budget}_{qkv_mode}_naive
#
# 注意：新增 input_mode / n_repeats / latency_ms_std / throughput_sps_std 字段
#   如果已有旧 expB.csv，建议删除后重跑（schema 不兼容新字段的 header）
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ── 日志 ──────────────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-eval_encoder/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/expB_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[log] → ${LOG_FILE}"

# ── 配置 ──────────────────────────────────────────────────────────────────────
TASKS="${TASKS:-cola sst2 mrpc qqp mnli qnli rte stsb}"
# 主表：固定 svd checkpoint，只比 backend（排除 method 对 rank 分配的干扰）
# 如需 rank-strategy sensitivity，覆盖 METHODS="svd fwsvd drone adasvd"
METHODS="${METHODS:-svd}"
BACKENDS="${BACKENDS:-naive sdpa flashsvd flashsvd15}"
# 双模式：real（真实数据，自然 padding）+ synthetic（全有效 token，0% padding）
INPUT_MODES="${INPUT_MODES:-real synthetic}"
# 重复测量次数（独立 warmup+measure，报 mean±std）
REPEAT="${REPEAT:-3}"
DTYPE="${DTYPE:-bf16}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WARMUP="${WARMUP:-10}"
MEASURE="${MEASURE:-50}"

# 与 expA / glue_pipeline.py 命名一致
RANK_ATTN="${RANK_ATTN:-48}"
RANK_FFN="${RANK_FFN:-256}"
RANK_WO="${RANK_WO:-208}"
QKV_MODE="${QKV_MODE:-per_head}"
BUDGET="${BUDGET:-0.527}"

MODEL_BASE_DIR="${MODEL_BASE_DIR:-eval_encoder/models}"
OUT_CSV="${OUT_CSV:-eval_encoder/eval_results/expB.csv}"
# 对齐检查：0=关（默认，不影响速度）；1=开（每个 job 多 load 一次 naive 做 logit diff）
# 用法：ALIGN=1 bash expB.sh
ALIGN="${ALIGN:-0}"
ALIGN_FLAG=""
[[ "${ALIGN}" -eq 1 ]] && ALIGN_FLAG="--check_alignment"

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

                python eval_encoder/scripts/analyze_compute.py \
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
echo "  完成  成功=${OK}  失败=${FAIL}  跳过=${SKIP}"
echo "  结果 → ${OUT_CSV}"
echo "══════════════════════════════════════════════════════════════════════"

[[ "${FAIL}" -eq 0 ]]
