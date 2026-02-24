#!/usr/bin/env bash
################################################################################
# run_sdpa_ablation.sh
#
# SDPA 消融实验：隔离"flash attention 收益"与"FlashSVD 投影融合收益"
#
# 背景：
#   Naive(einsum) vs FlashSVD 对比混淆了两个变量：
#     ① attention 计算（O(n²) einsum vs flash attention）
#     ② 低秩投影处理（独立 matmul vs fused Triton kernel）
#   加入 Naive(SDPA) 中间档，可拆分两者贡献：
#
#   Naive(einsum) → Naive(SDPA) → FlashSVD(Triton)
#     ~2003 MB        ~720 MB?       ~708 MB
#     ~190 sps        ~220 sps?      ~330 sps
#     ←── flash attn ──→←── proj fusion ──→
#
# 两种运行模式：
#   模式 A（推荐）：有已保存的压缩模型 → --load_model_dir 直接复用，跳过压缩
#   模式 B（备选）：无存档 → 重新压缩 + 测量（SVD 约 3.5 min/task）
#
# 用法：
#   # 默认（per-head ra48/rf256/rw208，8 tasks，自动检测是否有存档）
#   bash eval_encoder/scripts/run_sdpa_ablation.sh
#
#   # 强制重新压缩（忽略存档）
#   FORCE_RECOMPRESS=true bash eval_encoder/scripts/run_sdpa_ablation.sh
#
#   # 指定方法（需要先有对应存档，或 FORCE_RECOMPRESS=true）
#   METHODS="svd fwsvd drone adasvd" bash eval_encoder/scripts/run_sdpa_ablation.sh
#
# 如何先保存压缩模型（模式 A 的前提）：
#   METHODS="svd fwsvd drone adasvd" \
#   STAGES=no_finetune BACKENDS=naive \
#   QKV_MODE=per_head RANK_ATTN=48 RANK_FFN=256 RANK_WO=208 BUDGET=0.527 \
#   bash eval_encoder/scripts/compare_all_methods.sh
#   # 注：需在 one_click_glue.sh 里临时加 SAVE_MODEL=true，
#   #     或直接用下面的 FORCE_RECOMPRESS=true 模式。
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ─────────────────────────────── 压缩配置 ────────────────────────────────────
# 与主 benchmark 保持一致
METHODS="${METHODS:-svd fwsvd drone adasvd}"
TASKS="${TASKS:-cola sst2 mrpc qqp mnli qnli rte stsb}"

# Per-head 秩配置（与 benchmark_perhead_report.md 一致）
RANK_ATTN="${RANK_ATTN:-48}"
RANK_FFN="${RANK_FFN:-256}"
RANK_WO="${RANK_WO:-208}"
QKV_MODE="${QKV_MODE:-per_head}"
BUDGET="${BUDGET:-0.527}"                        # AdaSVD 专用
CALIB_BATCHES="${CALIB_BATCHES:-4}"             # FWSVD / DRONE 校准批次
ADASVD_CALIB_SAMPLES="${ADASVD_CALIB_SAMPLES:-4000}"  # AdaSVD ARS 校准样本数
ADASVD_STEPS="${ADASVD_STEPS:-800}"                    # AdaSVD ARS 训练步数

# 模型前缀（用于 HuggingFace task-specific 模型）
TASK_MODEL_PREFIX="${TASK_MODEL_PREFIX:-textattack}"

# ─────────────────────────────── 性能测量配置 ─────────────────────────────────
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DTYPE="${DTYPE:-fp32}"
FULL_VALIDATION="${FULL_VALIDATION:-true}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
MEASURE_STEPS="${MEASURE_STEPS:-50}"
NUM_RUNS="${NUM_RUNS:-1}"

# ─────────────────────────────── 路径配置 ────────────────────────────────────
MODEL_BASE_DIR="${MODEL_BASE_DIR:-eval_encoder/models}"
OUT_CSV="${OUT_CSV:-eval_encoder/eval_results/encoder_runs.csv}"
FORCE_RECOMPRESS="${FORCE_RECOMPRESS:-false}"

# ─────────────────────────────── 日志 ────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="eval_encoder/eval_results"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/sdpa_ablation_${TIMESTAMP}.log"

echo "════════════════════════════════════════════════════════════════════" | tee "${LOG_FILE}"
echo "SDPA Ablation  |  ${TIMESTAMP}" | tee -a "${LOG_FILE}"
echo "Methods:  ${METHODS}" | tee -a "${LOG_FILE}"
echo "Tasks:    ${TASKS}" | tee -a "${LOG_FILE}"
echo "Config:   rank_attn=${RANK_ATTN} rank_ffn=${RANK_FFN} rank_wo=${RANK_WO} qkv_mode=${QKV_MODE}" | tee -a "${LOG_FILE}"
echo "Perf:     seq_len=${SEQ_LEN} bs=${BATCH_SIZE} dtype=${DTYPE} full_validation=${FULL_VALIDATION}" | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════════════════════════════" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

SUCCESS=0
FAIL=0

# ─────────────────────────────── 辅助函数 ────────────────────────────────────

# 推导模型子目录名（与 glue_pipeline.py / run_encoder_benchmark.py 命名一致）
_model_subdir() {
    local method="$1"
    if [[ "${method}" == "adasvd" ]]; then
        echo "adasvd_b${BUDGET}_${QKV_MODE}_naive"
    else
        echo "${method}_ra${RANK_ATTN}_rf${RANK_FFN}_rw${RANK_WO}_${QKV_MODE}_naive"
    fi
}

# 拼接性能测量 flags
_perf_flags() {
    local flags="--warmup_steps ${WARMUP_STEPS} --measure_steps ${MEASURE_STEPS} --num_runs ${NUM_RUNS}"
    if [[ "${FULL_VALIDATION}" == "true" ]]; then
        flags="${flags} --full_validation"
    fi
    echo "${flags}"
}

# ─────────────────────────────── 主循环 ──────────────────────────────────────
for METHOD in ${METHODS}; do
    for TASK in ${TASKS}; do
        echo "────────────────────────────────────────────────────────────────────" | tee -a "${LOG_FILE}"
        echo "Method=${METHOD}  Task=${TASK}" | tee -a "${LOG_FILE}"
        echo "────────────────────────────────────────────────────────────────────" | tee -a "${LOG_FILE}"

        MODEL_SUBDIR="$(_model_subdir "${METHOD}")"
        MODEL_DIR="${MODEL_BASE_DIR}/${TASK}/${MODEL_SUBDIR}"

        # 决定模式
        if [[ "${FORCE_RECOMPRESS}" != "true" ]] && [[ -d "${MODEL_DIR}" ]]; then
            # ── 模式 A：复用已保存模型 ──────────────────────────────────────────
            echo "[mode=load] Using saved model: ${MODEL_DIR}" | tee -a "${LOG_FILE}"

            CMD=(
                python eval_encoder/run_encoder_benchmark.py
                --load_model_dir "${MODEL_DIR}"
                --task "${TASK}"
                --attn_mode sdpa
                --seq_len "${SEQ_LEN}"
                --batch_size "${BATCH_SIZE}"
                --dtype "${DTYPE}"
                --skip_eval
                --out_csv "${OUT_CSV}"
            )
            read -ra PERF_ARR <<< "$(_perf_flags)"
            CMD+=("${PERF_ARR[@]}")

        else
            # ── 模式 B：重新压缩（无存档或强制重压）──────────────────────────────
            if [[ "${FORCE_RECOMPRESS}" != "true" ]]; then
                echo "[mode=recompress] No saved model found at ${MODEL_DIR}, compressing fresh" | tee -a "${LOG_FILE}"
            else
                echo "[mode=recompress] FORCE_RECOMPRESS=true, recompressing" | tee -a "${LOG_FILE}"
            fi

            # task-specific model_id
            MODEL_ID="textattack/bert-base-uncased-$(echo "${TASK}" | tr '[:lower:]' '[:upper:]')"
            # Special cases
            case "${TASK}" in
                stsb) MODEL_ID="textattack/bert-base-uncased-STS-B" ;;
                mnli) MODEL_ID="textattack/bert-base-uncased-MNLI"  ;;
                cola) MODEL_ID="textattack/bert-base-uncased-CoLA"  ;;
                sst2) MODEL_ID="textattack/bert-base-uncased-SST-2" ;;
                mrpc) MODEL_ID="textattack/bert-base-uncased-MRPC"  ;;
                qqp)  MODEL_ID="textattack/bert-base-uncased-QQP"   ;;
                qnli) MODEL_ID="textattack/bert-base-uncased-QNLI"  ;;
                rte)  MODEL_ID="textattack/bert-base-uncased-RTE"   ;;
            esac

            CMD=(
                python eval_encoder/run_encoder_benchmark.py
                --model_id "${MODEL_ID}"
                --method "${METHOD}"
                --task "${TASK}"
                --qkv_mode  "${QKV_MODE}"
                --attn_mode sdpa
                --backend naive
                --seq_len "${SEQ_LEN}"
                --batch_size "${BATCH_SIZE}"
                --dtype "${DTYPE}"
                --skip_eval
                --out_csv "${OUT_CSV}"
            )

            if [[ "${METHOD}" == "adasvd" ]]; then
                # AdaSVD 用 budget，不用 rank_attn/ffn/wo
                CMD+=(
                    --budget "${BUDGET}"
                    --adasvd_calib_samples "${ADASVD_CALIB_SAMPLES:-4000}"
                    --adasvd_steps        "${ADASVD_STEPS:-800}"
                )
            else
                # SVD / FWSVD / DRONE 用 component-specific ranks
                CMD+=(
                    --rank_attn "${RANK_ATTN}"
                    --rank_ffn  "${RANK_FFN}"
                    --rank_wo   "${RANK_WO}"
                    --calib_batches "${CALIB_BATCHES}"
                )
            fi

            read -ra PERF_ARR <<< "$(_perf_flags)"
            CMD+=("${PERF_ARR[@]}")
        fi

        echo "CMD: ${CMD[*]}" | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"

        if "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"; then
            echo "✅ Done: method=${METHOD} task=${TASK}" | tee -a "${LOG_FILE}"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "❌ FAILED: method=${METHOD} task=${TASK}" | tee -a "${LOG_FILE}"
            FAIL=$((FAIL + 1))
        fi
        echo "" | tee -a "${LOG_FILE}"
    done
done

# ─────────────────────────────── 完成 ────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════" | tee -a "${LOG_FILE}"
echo "SDPA Ablation Complete" | tee -a "${LOG_FILE}"
echo "Success: ${SUCCESS}  Fail: ${FAIL}" | tee -a "${LOG_FILE}"
echo "Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "结果写入: ${OUT_CSV}" | tee -a "${LOG_FILE}"
echo "精度数值从对应 naive(einsum) 行复用，无需重新评估。" | tee -a "${LOG_FILE}"
echo "════════════════════════════════════════════════════════════════════" | tee -a "${LOG_FILE}"

if [[ "${FAIL}" -gt 0 ]]; then
    exit 1
fi
exit 0
