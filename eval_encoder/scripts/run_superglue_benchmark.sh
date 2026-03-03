#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_superglue_benchmark.sh
#
# 在 SuperGLUE / HANS / ANLI 上跑所有压缩方法 × 多个 backend 并汇总结果。
# 不进行微调，直接评估压缩后精度（zero-shot compression eval）。
#
# Checkpoint 逻辑：
#   压缩只做一次（naive backend），checkpoint 命名带 _naive 后缀。
#   backend 循环时从 naive checkpoint --load_model_dir，只换 --backend。
#
# 任务分类：
#   SuperGLUE-Core（进平均）: boolq, rte_sg, wic, copa
#   Diagnostic（不进平均）:   cb   [高方差，56例，仅参考]
#   Robustness:               hans, anli_r1, anli_r2, anli_r3
#
# COPA 说明：
#   使用 NLI two-choice scoring（非标准分类）。
#   模型：textattack/bert-base-uncased-MNLI（class 1 = entailment）。
#   校准：使用 copa 自带的 train split（400 例）。
#
# CB 说明：
#   验证集仅 56 例，高方差诊断任务，不计入 SuperGLUE 平均。
#   结果仅供参考，建议搭配多 seed 解读。
#
# 覆盖方法（5 个）：
#   dense, svd, fwsvd, drone, adasvd
#
# 用法：
#   cd lowrankarena/
#   bash eval_encoder/scripts/run_superglue_benchmark.sh
#
# 可覆盖变量示例：
#   TASKS="boolq hans copa"
#   METHODS="svd fwsvd"
#   BACKENDS="naive sdpa flashsvd"    # fp32 下不含 flashsvd15（会 cast 到 fp16）
#   DTYPE=fp32 BACKENDS="naive sdpa flashsvd"   # fp32 精度测试（不含 flashsvd15）
#   RECOMPRESS=true     # 强制重新压缩（忽略已有 checkpoint）
#   OUT_CSV=eval_encoder/eval_results/expA.csv
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ── 日志 ─────────────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-eval_encoder/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/superglue_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[log] → ${LOG_FILE}"

# ── 配置 ─────────────────────────────────────────────────────────────────────
TASKS="${TASKS:-boolq rte_sg wic copa cb hans anli_r1 anli_r2 anli_r3}"
METHODS="${METHODS:-dense svd fwsvd drone adasvd}"
BACKENDS="${BACKENDS:-naive sdpa flashsvd flashsvd15}"

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
BATCH_SIZE="${BATCH_SIZE:-32}"

MODEL_BASE_DIR="${MODEL_BASE_DIR:-eval_encoder/models}"
OUT_CSV="${OUT_CSV:-eval_encoder/eval_results/expA.csv}"

# RECOMPRESS=true 强制忽略已有 checkpoint，重新压缩
RECOMPRESS="${RECOMPRESS:-false}"

# ── task → model_id 映射 ──────────────────────────────────────────────────────
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

# task 是否需要跨任务校准（train_split=None）
_calib_task_for() {
    case "$1" in
        hans) echo "mnli" ;;   # hans 无 train split
        *)    echo "" ;;
    esac
}

# checkpoint 子目录名 — 始终带 _naive 后缀（压缩统一用 naive 保存）
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

        # ── Step 1：确保 naive checkpoint 存在（dense 不需要）────────────────
        if [[ "${METHOD}" != "dense" ]]; then
            if [[ "${RECOMPRESS}" == "true" || ! -d "${CKPT_DIR}" ]]; then
                echo "── ${TASK} / ${METHOD}  [compress → ${CKPT_DIR}]"

                if [[ "${METHOD}" == "adasvd" ]]; then
                    python eval_encoder/run_encoder_benchmark.py \
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
                    python eval_encoder/run_encoder_benchmark.py \
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

        # ── Step 2：遍历其余 backend（从 naive checkpoint load）──────────────
        # sdpa 实际上是 --backend naive --attn_mode sdpa，不是独立 backend
        for BACKEND in ${BACKENDS}; do
            # naive 在 Step 1 压缩时已经评估并写 CSV，跳过重复跑
            [[ "${BACKEND}" == "naive" && "${METHOD}" != "dense" && -d "${CKPT_DIR}" && "${RECOMPRESS}" != "true" ]] && \
                echo "   [skip naive — already in CSV from compress step]" && continue

            # 翻译 sdpa → --backend naive --attn_mode sdpa
            if [[ "${BACKEND}" == "sdpa" ]]; then
                BACKEND_ARGS=(--backend naive --attn_mode sdpa)
            else
                BACKEND_ARGS=(--backend "${BACKEND}")
            fi

            echo "   ── ${TASK} / ${METHOD} / ${BACKEND}"

            if [[ "${METHOD}" == "dense" ]]; then
                python eval_encoder/run_encoder_benchmark.py \
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
                python eval_encoder/run_encoder_benchmark.py \
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
echo "  完成  成功=${OK}  失败=${FAIL}"
echo "  结果 → ${OUT_CSV}"
echo "══════════════════════════════════════════════════════════════════════"

[[ "${FAIL}" -eq 0 ]]
