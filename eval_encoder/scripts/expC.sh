#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# expC.sh  —  Scaling Experiment
#
# 固定同一 checkpoint，量化随 seq_len / batch_size 变化的
# throughput / latency / peak_mem 趋势（4 个 backend 对比）。
#
# 设计原则
# ────────────────────────────────────────────────────────────────────────────
#   • 复用 expA 产物（不重新压缩）；缺 checkpoint 会 [skip]，不 abort
#   • 只选 2 个代表方法：svd（均匀 rank）+ adasvd（异质 rank）
#   • 只选 2 个代表任务：mnli（长句）+ stsb（短句，padding 特征不同）
#   • 内部直接调 analyze_compute.py（与 expB 一致，不引入新口径）
#   • 两个独立 phase：
#       seqlen — batch=32 固定，sweep seq_len = 128 256 384 512
#       batch  — seq=512 固定，sweep batch_size = 8 16 32 64
#   • input_mode=synthetic（0% padding，与 expB 一致，公平比较 backend）
#   • repeat=3（mean±std 控制方差，与 expB 一致）
#   • 每次运行前清空对应 CSV（避免 schema 不兼容的旧行混入）
#
# ⚠️ dense 已从默认 METHODS 移除：dense 模型路径命名与 SVD 方法不同，
#    需要额外处理。如需 dense baseline，请用 run_encoder_benchmark.py。
#
# ⚠️ seq_len=768 暂不支持：BERT-base max_position_embeddings=512，
#    超出会报错。ModernBERT 支持更长序列，但 expC 目前只跑 BERT-base。
#
# 用法
# ────────────────────────────────────────────────────────────────────────────
#   cd lowrankarena/
#   bash eval_encoder/scripts/expC.sh                  # 全量 (seqlen + batch)
#   PHASES=seqlen bash eval_encoder/scripts/expC.sh    # 只跑 seq_len sweep
#   PHASES=batch  bash eval_encoder/scripts/expC.sh    # 只跑 batch sweep
#
# 可覆盖变量
#   PHASES="seqlen batch"
#   TASKS="mnli stsb"
#   METHODS="svd adasvd"            # dense 已从默认值移除（路径命名不同）
#   BACKENDS="naive sdpa flashsvd flashsvd15"
#   SEQ_LENS="128 256 384 512"      # seqlen phase 的扫描点
#   BATCH_SIZES="8 16 32 64"        # batch phase 的扫描点
#   BATCH_FIXED=32                  # seqlen phase 固定 batch
#   SEQ_FIXED=512                   # batch phase 固定 seq_len
#   DTYPE=bf16
#   INPUT_MODE=synthetic            # real | synthetic（默认 synthetic，0% padding）
#   REPEAT=3                        # 重复测量次数（mean±std，与 expB 一致）
#   WARMUP=10  MEASURE=50           # 与 expB 校准口径一致
#   ALIGN=0                         # 0=关（默认），1=开（加 --check_alignment，记录 logit_max_diff）
#   MODEL_BASE_DIR=eval_encoder/models
#   OUT_SEQLEN=eval_encoder/eval_results/expC_seqlen.csv
#   OUT_BATCH=eval_encoder/eval_results/expC_batch.csv
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ── 日志 ──────────────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-eval_encoder/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/expC_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[log] → ${LOG_FILE}"

# ── 配置 ──────────────────────────────────────────────────────────────────────
PHASES="${PHASES:-seqlen batch}"
TASKS="${TASKS:-mnli stsb}"
# dense 已移除：其模型路径命名与 SVD checkpoint 不同，会导致路径查找失败
METHODS="${METHODS:-svd adasvd}"
BACKENDS="${BACKENDS:-naive sdpa flashsvd flashsvd15}"
DTYPE="${DTYPE:-bf16}"
WARMUP="${WARMUP:-10}"
MEASURE="${MEASURE:-50}"
# synthetic: 0% padding，消除 padding 率对 backend 吞吐的干扰（与 expB 一致）
INPUT_MODE="${INPUT_MODE:-synthetic}"
# 独立重复次数：mean±std（与 expB 校准口径一致）
REPEAT="${REPEAT:-3}"

# seqlen phase 参数
SEQ_LENS="${SEQ_LENS:-128 256 384 512}"
BATCH_FIXED="${BATCH_FIXED:-32}"

# batch phase 参数
BATCH_SIZES="${BATCH_SIZES:-8 16 32 64}"
SEQ_FIXED="${SEQ_FIXED:-512}"

# checkpoint 命名（与 expA / glue_pipeline.py 一致）
RANK_ATTN="${RANK_ATTN:-48}"
RANK_FFN="${RANK_FFN:-256}"
RANK_WO="${RANK_WO:-208}"
QKV_MODE="${QKV_MODE:-per_head}"
BUDGET="${BUDGET:-0.527}"

MODEL_BASE_DIR="${MODEL_BASE_DIR:-eval_encoder/models}"
OUT_SEQLEN="${OUT_SEQLEN:-eval_encoder/eval_results/expC_seqlen.csv}"
OUT_BATCH="${OUT_BATCH:-eval_encoder/eval_results/expC_batch.csv}"
# 对齐检查：0=关（默认）；1=开（每个 job 多 load 一次 naive 做 logit diff）
# 用法：ALIGN=1 bash expC.sh
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

# flashsvd15 仅支持 bf16 / fp16
_skip_backend() {
    local backend="$1"
    if [[ "${backend}" == "flashsvd15" && "${DTYPE}" == "fp32" ]]; then
        echo "true"
    else
        echo "false"
    fi
}

# ── 内部：对一个 (method, task, backend) 组合跑 analyze_compute.py ─────────────
# 返回值：0=成功  2=跳过（checkpoint 不存在 / backend 不支持）  其他=失败
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

    python eval_encoder/scripts/analyze_compute.py \
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

    # 清空旧 CSV，避免 schema 不兼容的历史行混入
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

    # 清空旧 CSV，避免 schema 不兼容的历史行混入
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
echo "  expC 完成  成功=${OK}  失败=${FAIL}  跳过=${SKIP}"
echo ""
echo "  输出："
[[ "${PHASES}" == *"seqlen"* ]] && echo "    seq_len sweep → ${OUT_SEQLEN}"
[[ "${PHASES}" == *"batch"*  ]] && echo "    batch sweep   → ${OUT_BATCH}"
echo "══════════════════════════════════════════════════════════════════════"

[[ "${FAIL}" -eq 0 ]]
