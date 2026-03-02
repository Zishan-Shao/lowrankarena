#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_superglue_benchmark.sh
#
# 在 SuperGLUE / HANS / ANLI 上跑所有压缩方法并汇总结果。
# 不进行微调，直接评估压缩后的精度（zero-shot compression eval）。
#
# Checkpoint 逻辑：
#   若 MODEL_BASE_DIR/{task}/{method_subdir}/ 存在 → --load_model_dir（跳过压缩）
#   否则 → 重新压缩，并 --save_model --save_dir MODEL_BASE_DIR/{task}
#
# 覆盖任务（8 个）：
#   boolq, cb, rte_sg, wic, hans, anli_r1, anli_r2, anli_r3
#
# 覆盖方法（5 个）：
#   dense, svd, fwsvd, drone, adasvd
#
# 用法：
#   cd lowrankarena/
#   bash eval_encoder/scripts/run_superglue_benchmark.sh
#
# 可覆盖变量示例：
#   TASKS="boolq hans"        METHODS="svd fwsvd"
#   RANK_ATTN=64 RANK_FFN=256 RANK_WO=256
#   DTYPE=bf16  SEQ_LEN=256   BACKEND=naive
#   RECOMPRESS=true           # 强制重新压缩（忽略已有 checkpoint）
#   OUT_CSV=eval_encoder/eval_results/superglue_results.csv
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ── 配置 ─────────────────────────────────────────────────────────────────────
TASKS="${TASKS:-boolq cb rte_sg wic hans anli_r1 anli_r2 anli_r3}"
METHODS="${METHODS:-dense svd fwsvd drone adasvd}"

# SVD / FWSVD / DRONE 的 rank（与 GLUE expA 完全一致）
RANK_ATTN="${RANK_ATTN:-48}"
RANK_FFN="${RANK_FFN:-256}"
RANK_WO="${RANK_WO:-208}"
QKV_MODE="${QKV_MODE:-per_head}"

# AdaSVD budget（0.527 ≈ 与上面 rank 等参数量）
BUDGET="${BUDGET:-0.527}"
ADASVD_CALIB_SAMPLES="${ADASVD_CALIB_SAMPLES:-4000}"
ADASVD_STEPS="${ADASVD_STEPS:-800}"

CALIB_BATCHES="${CALIB_BATCHES:-16}"
DTYPE="${DTYPE:-bf16}"
SEQ_LEN="${SEQ_LEN:-512}"
BACKEND="${BACKEND:-naive}"
BATCH_SIZE="${BATCH_SIZE:-32}"

MODEL_BASE_DIR="${MODEL_BASE_DIR:-eval_encoder/models}"
OUT_CSV="${OUT_CSV:-eval_encoder/eval_results/superglue_results.csv}"

# RECOMPRESS=true 可强制跳过 checkpoint 直接重新压缩
RECOMPRESS="${RECOMPRESS:-false}"

# ── task → model_id 映射 ──────────────────────────────────────────────────────
_model_id_for_task() {
    case "$1" in
        boolq)   echo "howey/bert-base-uncased-boolq" ;;
        cb)      echo "textattack/bert-base-uncased-MNLI" ;;
        rte_sg)  echo "howey/bert-base-uncased-rte" ;;
        wic)     echo "rycecorn/Bert-fine-tuned-WiC" ;;
        hans)    echo "textattack/bert-base-uncased-MNLI" ;;
        anli_r1) echo "textattack/bert-base-uncased-MNLI" ;;
        anli_r2) echo "textattack/bert-base-uncased-MNLI" ;;
        anli_r3) echo "textattack/bert-base-uncased-MNLI" ;;
        *)       echo "" ;;
    esac
}

# task 是否需要跨任务校准（train_split=None）
_calib_task_for() {
    case "$1" in
        hans) echo "mnli" ;;   # hans 无 train split
        *)    echo "" ;;
    esac
}

# checkpoint 子目录名（与 glue_pipeline.py / run_encoder_benchmark.py 保存命名一致）
_model_subdir() {
    local method="$1"
    case "${method}" in
        dense)   echo "dense_naive" ;;
        adasvd)  echo "adasvd_b${BUDGET}_${QKV_MODE}_${BACKEND}" ;;
        *)       echo "${method}_ra${RANK_ATTN}_rf${RANK_FFN}_rw${RANK_WO}_${QKV_MODE}_${BACKEND}" ;;
    esac
}

echo "══════════════════════════════════════════════════════════════════════"
echo "  SuperGLUE / HANS / ANLI Compression Benchmark"
echo "  tasks:   ${TASKS}"
echo "  methods: ${METHODS}"
echo "  ranks:   ra${RANK_ATTN}_rf${RANK_FFN}_rw${RANK_WO}  qkv=${QKV_MODE}"
echo "  adasvd:  budget=${BUDGET}  calib_samples=${ADASVD_CALIB_SAMPLES}  steps=${ADASVD_STEPS}"
echo "  dtype:   ${DTYPE}   seq_len: ${SEQ_LEN}   backend: ${BACKEND}"
echo "  models:  ${MODEL_BASE_DIR}  recompress: ${RECOMPRESS}"
echo "  out_csv: ${OUT_CSV}"
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

    for METHOD in ${METHODS}; do
        SUBDIR="$(_model_subdir "${METHOD}")"
        CKPT_DIR="${MODEL_BASE_DIR}/${TASK}/${SUBDIR}"

        echo "── ${TASK} / ${METHOD}"

        # ── dense ─────────────────────────────────────────────────────────
        if [[ "${METHOD}" == "dense" ]]; then
            python eval_encoder/run_encoder_benchmark.py \
                --task       "${TASK}" \
                --model_id   "${MODEL_ID}" \
                --method     dense \
                --backend    "${BACKEND}" \
                --dtype      "${DTYPE}" \
                --seq_len    "${SEQ_LEN}" \
                --batch_size "${BATCH_SIZE}" \
                --out_csv    "${OUT_CSV}" \
                && OK=$((OK + 1)) || FAIL=$((FAIL + 1))
            continue
        fi

        # ── SVD-based + AdaSVD ────────────────────────────────────────────
        EXTRA_ARGS=()
        [[ -n "${CALIB_TASK}" ]] && EXTRA_ARGS+=(--calib_task "${CALIB_TASK}")

        if [[ "${RECOMPRESS}" != "true" && -d "${CKPT_DIR}" ]]; then
            # checkpoint 已存在，直接 load
            echo "   [load] ${CKPT_DIR}"
            python eval_encoder/run_encoder_benchmark.py \
                --task           "${TASK}" \
                --model_id       "${MODEL_ID}" \
                --method         "${METHOD}" \
                --load_model_dir "${CKPT_DIR}" \
                --backend        "${BACKEND}" \
                --dtype          "${DTYPE}" \
                --seq_len        "${SEQ_LEN}" \
                --batch_size     "${BATCH_SIZE}" \
                --out_csv        "${OUT_CSV}" \
                "${EXTRA_ARGS[@]}" \
                && OK=$((OK + 1)) || FAIL=$((FAIL + 1))
        elif [[ "${METHOD}" == "adasvd" ]]; then
            # AdaSVD：用 budget，不用 rank 参数
            echo "   [compress] adasvd budget=${BUDGET}  saving to ${CKPT_DIR}"
            python eval_encoder/run_encoder_benchmark.py \
                --task                 "${TASK}" \
                --model_id             "${MODEL_ID}" \
                --method               adasvd \
                --budget               "${BUDGET}" \
                --qkv_mode             "${QKV_MODE}" \
                --adasvd_calib_samples "${ADASVD_CALIB_SAMPLES}" \
                --adasvd_steps         "${ADASVD_STEPS}" \
                --backend              "${BACKEND}" \
                --dtype                "${DTYPE}" \
                --seq_len              "${SEQ_LEN}" \
                --batch_size           "${BATCH_SIZE}" \
                --save_model \
                --save_dir             "${MODEL_BASE_DIR}/${TASK}" \
                --out_csv              "${OUT_CSV}" \
                "${EXTRA_ARGS[@]}" \
                && OK=$((OK + 1)) || FAIL=$((FAIL + 1))
        else
            # SVD / FWSVD / DRONE：用 rank 参数
            echo "   [compress] saving to ${CKPT_DIR}"
            python eval_encoder/run_encoder_benchmark.py \
                --task          "${TASK}" \
                --model_id      "${MODEL_ID}" \
                --method        "${METHOD}" \
                --rank_attn     "${RANK_ATTN}" \
                --rank_ffn      "${RANK_FFN}" \
                --rank_wo       "${RANK_WO}" \
                --qkv_mode      "${QKV_MODE}" \
                --calib_batches "${CALIB_BATCHES}" \
                --backend       "${BACKEND}" \
                --dtype         "${DTYPE}" \
                --seq_len       "${SEQ_LEN}" \
                --batch_size    "${BATCH_SIZE}" \
                --save_model \
                --save_dir      "${MODEL_BASE_DIR}/${TASK}" \
                --out_csv       "${OUT_CSV}" \
                "${EXTRA_ARGS[@]}" \
                && OK=$((OK + 1)) || FAIL=$((FAIL + 1))
        fi
    done
    echo ""
done

echo "══════════════════════════════════════════════════════════════════════"
echo "  完成  成功=${OK}  失败=${FAIL}"
echo "  结果 → ${OUT_CSV}"
echo "══════════════════════════════════════════════════════════════════════"

[[ "${FAIL}" -eq 0 ]]
