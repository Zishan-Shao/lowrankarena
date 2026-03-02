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
#   METHODS="svd adasvd"
#   BACKENDS="naive sdpa flashsvd flashsvd15"
#   SEQ_LENS="128 256 384 512"   # seqlen phase 的扫描点
#   BATCH_SIZES="8 16 32 64"     # batch phase 的扫描点
#   BATCH_FIXED=32               # seqlen phase 固定 batch
#   SEQ_FIXED=512                # batch phase 固定 seq_len
#   DTYPE=bf16
#   WARMUP=10  MEASURE=50
#   MODEL_BASE_DIR=eval_encoder/models
#   OUT_SEQLEN=eval_encoder/eval_results/expC_seqlen.csv
#   OUT_BATCH=eval_encoder/eval_results/expC_batch.csv
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ── 配置 ──────────────────────────────────────────────────────────────────────
PHASES="${PHASES:-seqlen batch}"
TASKS="${TASKS:-mnli stsb}"
METHODS="${METHODS:-svd adasvd dense}"   # dense 作 sanity baseline
BACKENDS="${BACKENDS:-naive sdpa flashsvd15}"
DTYPE="${DTYPE:-bf16}"
WARMUP="${WARMUP:-5}"
MEASURE="${MEASURE:-16}"   # 16 × bs=32 = 512 calib samples

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
_run_one() {
    local method="$1" task="$2" backend="$3" seq_len="$4" batch_size="$5" out_csv="$6"

    local subdir model_dir
    subdir="$(_model_subdir "${method}")"
    model_dir="${MODEL_BASE_DIR}/${task}/${subdir}"

    if [[ ! -d "${model_dir}" ]]; then
        echo "      [skip] Checkpoint not found: ${model_dir}"
        return 0
    fi

    if [[ "$(_skip_backend "${backend}")" == "true" ]]; then
        echo "      [skip] ${backend} requires bf16/fp16, current dtype=${DTYPE}"
        return 0
    fi

    python eval_encoder/scripts/analyze_compute.py \
        --model_dir  "${model_dir}" \
        --task       "${task}" \
        --backend    "${backend}" \
        --dtype      "${DTYPE}" \
        --seq_len    "${seq_len}" \
        --batch_size "${batch_size}" \
        --warmup     "${WARMUP}" \
        --measure    "${MEASURE}" \
        --out_csv    "${out_csv}"
}

# ── Banner ────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
echo "  expC — Scaling Experiment"
echo "  phases:   ${PHASES}"
echo "  tasks:    ${TASKS}   methods: ${METHODS}"
echo "  backends: ${BACKENDS}   dtype: ${DTYPE}"
echo "  config:   ra${RANK_ATTN}_rf${RANK_FFN}_rw${RANK_WO}  qkv=${QKV_MODE}  budget=${BUDGET}"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

OK=0
FAIL=0

# ── Phase A: seq_len sweep ─────────────────────────────────────────────────────
if [[ "${PHASES}" == *"seqlen"* ]]; then
    echo "══ Phase A: seq_len sweep (batch=${BATCH_FIXED}) ══════════════════"
    echo "   seq_lens: ${SEQ_LENS}"
    echo "   out_csv:  ${OUT_SEQLEN}"
    echo ""

    for SEQ_LEN in ${SEQ_LENS}; do
        for TASK in ${TASKS}; do
            echo "  seq_len=${SEQ_LEN}  task=${TASK}"
            for METHOD in ${METHODS}; do
                for BACKEND in ${BACKENDS}; do
                    echo "    → ${METHOD} / ${BACKEND}"
                    _run_one "${METHOD}" "${TASK}" "${BACKEND}" \
                             "${SEQ_LEN}" "${BATCH_FIXED}" "${OUT_SEQLEN}" \
                    && OK=$((OK + 1)) || { FAIL=$((FAIL + 1)); echo "    [FAILED] ${METHOD}/${TASK}/${BACKEND}/seq${SEQ_LEN}"; }
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

    for BATCH_SIZE in ${BATCH_SIZES}; do
        for TASK in ${TASKS}; do
            echo "  batch=${BATCH_SIZE}  task=${TASK}"
            for METHOD in ${METHODS}; do
                for BACKEND in ${BACKENDS}; do
                    echo "    → ${METHOD} / ${BACKEND}"
                    _run_one "${METHOD}" "${TASK}" "${BACKEND}" \
                             "${SEQ_FIXED}" "${BATCH_SIZE}" "${OUT_BATCH}" \
                    && OK=$((OK + 1)) || { FAIL=$((FAIL + 1)); echo "    [FAILED] ${METHOD}/${TASK}/${BACKEND}/bs${BATCH_SIZE}"; }
                done
            done
        done
        echo ""
    done
fi

echo "══════════════════════════════════════════════════════════════════════"
echo "  expC 完成  成功=${OK}  失败=${FAIL}"
echo ""
echo "  输出："
[[ "${PHASES}" == *"seqlen"* ]] && echo "    seq_len sweep → ${OUT_SEQLEN}"
[[ "${PHASES}" == *"batch"*  ]] && echo "    batch sweep   → ${OUT_BATCH}"
echo "══════════════════════════════════════════════════════════════════════"

[[ "${FAIL}" -eq 0 ]]
