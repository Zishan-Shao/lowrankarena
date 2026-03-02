#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_expA_sdpa.sh
#
# 补充 SDPA 消融行到 expA_backend.csv。
#
# 目的：拆分 FlashSVD 加速来源：
#   naive(einsum) → naive(SDPA) : Flash Attention 本身的收益
#   naive(SDPA)   → flashsvd15  : Triton 低秩投影融合的额外收益
#
# 新增行（backend=sdpa）：
#   SVD / AdaSVD × MNLI （默认）
#
# 公平性注意：
#   SVD 是确定性压缩，sdpa 行可直接与现有 naive 行对比。
#   AdaSVD 是随机的（rank 分布随校准数据浮动），所以本脚本对 adasvd
#   同时重跑 naive 和 sdpa 两行（同一次压缩），覆盖 CSV 中旧的 naive 行
#   以保证对比公平（同一 rank 分布下 einsum vs flash attention）。
#
# 用法：
#   cd lowrankarena/
#   bash eval_encoder/scripts/run_expA_sdpa.sh
#
# 可覆盖变量：
#   TASKS="mnli stsb"
#   METHODS="svd adasvd"
#   DTYPE=bf16  SEQ_LEN=512  BATCH_SIZE=32
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ── 配置（与 expA_backend.csv 保持一致）──────────────────────────────────────
TASKS="${TASKS:-mnli}"
METHODS="${METHODS:-svd adasvd}"
DTYPE="${DTYPE:-bf16}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WARMUP="${WARMUP:-10}"
MEASURE="${MEASURE:-50}"

# QKV 秩配置（与 expA_backend.csv 中现有行完全一致）
RANK_ATTN="${RANK_ATTN:-48}"
RANK_FFN="${RANK_FFN:-256}"
RANK_WO="${RANK_WO:-208}"
QKV_MODE="${QKV_MODE:-per_head}"
BUDGET="${BUDGET:-0.527}"   # AdaSVD only

OUT_CSV="${OUT_CSV:-eval_encoder/eval_results/expA_backend.csv}"

declare -A TASK_MODEL_IDS=(
    [mnli]="textattack/bert-base-uncased-MNLI"
    [stsb]="textattack/bert-base-uncased-STS-B"
    [cola]="textattack/bert-base-uncased-CoLA"
    [sst2]="textattack/bert-base-uncased-SST-2"
    [mrpc]="textattack/bert-base-uncased-MRPC"
    [qqp]="textattack/bert-base-uncased-QQP"
    [qnli]="textattack/bert-base-uncased-QNLI"
    [rte]="textattack/bert-base-uncased-RTE"
)

echo "══════════════════════════════════════════════════════════════"
echo "  expA SDPA 消融实验"
echo "  tasks=${TASKS}  methods=${METHODS}"
echo "  dtype=${DTYPE}  seq_len=${SEQ_LEN}  bs=${BATCH_SIZE}"
echo "  rank_attn=${RANK_ATTN} rank_ffn=${RANK_FFN} rank_wo=${RANK_WO} qkv_mode=${QKV_MODE}"
echo "  out_csv=${OUT_CSV}"
echo "══════════════════════════════════════════════════════════════"

_run_one() {
    local method="$1" task="$2" backend="$3"
    local model_id="${TASK_MODEL_IDS[$task]:-textattack/bert-base-uncased-$(echo "$task" | tr '[:lower:]' '[:upper:]')}"

    echo ""
    echo "── ${method} / ${task} / ${backend} ──"

    local base_args=(
        python eval_encoder/scripts/analyze_compute.py
        --method     "${method}"
        --task       "${task}"
        --model_id   "${model_id}"
        --backend    "${backend}"
        --dtype      "${DTYPE}"
        --seq_len    "${SEQ_LEN}"
        --batch_size "${BATCH_SIZE}"
        --warmup     "${WARMUP}"
        --measure    "${MEASURE}"
        --out_csv    "${OUT_CSV}"
    )

    if [[ "${method}" == "adasvd" ]]; then
        "${base_args[@]}" \
            --budget   "${BUDGET}" \
            --qkv_mode "${QKV_MODE}"
    else
        "${base_args[@]}" \
            --rank_attn "${RANK_ATTN}" \
            --rank_ffn  "${RANK_FFN}" \
            --rank_wo   "${RANK_WO}" \
            --qkv_mode  "${QKV_MODE}"
    fi
}

for TASK in ${TASKS}; do
    for METHOD in ${METHODS}; do
        if [[ "${METHOD}" == "adasvd" ]]; then
            # AdaSVD 是随机压缩：同一次压缩同时跑 naive 和 sdpa，
            # 覆盖旧行，保证两者使用完全相同的 rank 分布。
            _run_one "${METHOD}" "${TASK}" naive
            _run_one "${METHOD}" "${TASK}" sdpa
        else
            # SVD/FWSVD/DRONE 是确定性压缩：只跑 sdpa，
            # 直接与 CSV 中现有的 naive 行对比即可。
            _run_one "${METHOD}" "${TASK}" sdpa
        fi
    done
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  SDPA 消融完成 → ${OUT_CSV}"
echo "  新增：backend=sdpa 行；adasvd naive 行已刷新（同一次压缩）"
echo "══════════════════════════════════════════════════════════════"
