#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# expD.sh  —  Kernel-Level Analysis (nsys)
#
# 对 4 个代表点做 NVIDIA Nsight Systems profiling，量化：
#   - GEMM kernel 变体数（反映 rank 异质性）
#   - Triton fused kernel 占比 vs cuBLAS GEMM 占比
#   - kernel 碎片化程度（fragmentation_ratio）
#
# 4 个 profiling 点（与 plot_nsys_kernel.py 硬编码的 POINT_META 一致）
# ────────────────────────────────────────────────────────────────────────────
#   mnli_svd_naive          SVD     + naive
#   mnli_svd_flashsvd15     SVD     + flashsvd15
#   mnli_adasvd_naive       AdaSVD  + naive
#   mnli_adasvd_flashsvd15  AdaSVD  + flashsvd15
#
# 依赖
# ────────────────────────────────────────────────────────────────────────────
#   nsys (NVIDIA Nsight Systems CLI)  — 通常随 CUDA toolkit 安装
#   nsys stats —report cuda_gpu_kern_sum  — 提取 kernel summary
#   eval_encoder/scripts/parse_nsys_summary.py — 解析 summary → CSV
#   eval_encoder/scripts/plot_nsys_kernel.py   — 从 CSV 生成图
#
# ⚠️ 前提：expA 已生成下列 checkpoint（缺一则对应点 skip）：
#   eval_encoder/models/mnli/svd_ra48_rf256_rw208_per_head_naive/
#   eval_encoder/models/mnli/adasvd_b0.527_per_head_naive/
#
# 用法
# ────────────────────────────────────────────────────────────────────────────
#   cd lowrankarena/
#   bash eval_encoder/scripts/expD.sh
#
# 可覆盖变量
#   TASK=mnli
#   DTYPE=bf16
#   SEQ_LEN=512
#   BATCH_SIZE=32
#   INPUT_MODE=synthetic  # real | synthetic（默认 synthetic，kernel attribution 需要 0% padding）
#   WARMUP=5              # nsys warmup（少几步，省时间）
#   MEASURE=50            # nsys measure（50 步采样足够的 kernel 多样性）
#   MODEL_BASE_DIR=eval_encoder/models
#   NSYS_DIR=eval_encoder/eval_results/nsys   # .nsys-rep 文件输出目录
#   SUMMARY_TXT=eval_encoder/eval_results/nsys/nsys_summary.txt
#   OUT_CSV=eval_encoder/eval_results/expD.csv
#   FIGURES_DIR=eval_encoder/eval_results/figures
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ── 日志 ──────────────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-eval_encoder/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/expD_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[log] → ${LOG_FILE}"

# ── 配置 ──────────────────────────────────────────────────────────────────────
TASK="${TASK:-mnli}"
DTYPE="${DTYPE:-bf16}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WARMUP="${WARMUP:-5}"
MEASURE="${MEASURE:-50}"   # 50 步：足够采样 kernel 多样性，与 expB/expC 校准口径一致
# synthetic: 0% padding，kernel attribution 需要 fully-utilized input（实际 padding 会稀释 GEMM 贡献）
INPUT_MODE="${INPUT_MODE:-synthetic}"

RANK_ATTN="${RANK_ATTN:-48}"
RANK_FFN="${RANK_FFN:-256}"
RANK_WO="${RANK_WO:-208}"
QKV_MODE="${QKV_MODE:-per_head}"
BUDGET="${BUDGET:-0.527}"

MODEL_BASE_DIR="${MODEL_BASE_DIR:-eval_encoder/models}"
NSYS_DIR="${NSYS_DIR:-eval_encoder/eval_results/nsys}"
SUMMARY_TXT="${SUMMARY_TXT:-${NSYS_DIR}/nsys_summary.txt}"
OUT_CSV="${OUT_CSV:-eval_encoder/eval_results/expD.csv}"
FIGURES_DIR="${FIGURES_DIR:-eval_encoder/eval_results/figures}"

# ── 检查 nsys ─────────────────────────────────────────────────────────────────
if ! command -v nsys &>/dev/null; then
    echo "[error] nsys not found. Install NVIDIA Nsight Systems:"
    echo "        https://developer.nvidia.com/nsight-systems"
    exit 1
fi

mkdir -p "${NSYS_DIR}" "${FIGURES_DIR}"

# ── profiling 点定义（与 plot_nsys_kernel.py POINT_META 一致）─────────────────
# 格式：POINT_TAG:METHOD:BACKEND
POINTS=(
    "${TASK}_svd_naive:svd:naive"
    "${TASK}_svd_flashsvd15:svd:flashsvd15"
    "${TASK}_adasvd_naive:adasvd:naive"
    "${TASK}_adasvd_flashsvd15:adasvd:flashsvd15"
)

# ── checkpoint 子目录名 ────────────────────────────────────────────────────────
_model_subdir() {
    local method="$1"
    if [[ "${method}" == "adasvd" ]]; then
        echo "adasvd_b${BUDGET}_${QKV_MODE}_naive"
    else
        echo "${method}_ra${RANK_ATTN}_rf${RANK_FFN}_rw${RANK_WO}_${QKV_MODE}_naive"
    fi
}

echo "══════════════════════════════════════════════════════════════════════"
echo "  expD — Kernel-Level Analysis (nsys)"
echo "  task:       ${TASK}   dtype: ${DTYPE}   seq_len: ${SEQ_LEN}   bs: ${BATCH_SIZE}"
echo "  warmup:     ${WARMUP}   measure: ${MEASURE}"
echo "  input_mode: ${INPUT_MODE}"
echo "  nsys_dir: ${NSYS_DIR}"
echo "  summary:  ${SUMMARY_TXT}"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

# 清空（重新生成）summary txt
> "${SUMMARY_TXT}"

OK=0
FAIL=0
SKIP=0

# ── 主循环：profile 每个点 ────────────────────────────────────────────────────
for ENTRY in "${POINTS[@]}"; do
    POINT_TAG="${ENTRY%%:*}"                  # e.g. mnli_svd_naive
    REST="${ENTRY#*:}"
    METHOD="${REST%%:*}"                      # e.g. svd
    BACKEND="${REST#*:}"                      # e.g. naive

    SUBDIR="$(_model_subdir "${METHOD}")"
    MODEL_DIR="${MODEL_BASE_DIR}/${TASK}/${SUBDIR}"

    echo "── ${POINT_TAG}  (${MODEL_DIR})"

    if [[ ! -d "${MODEL_DIR}" ]]; then
        echo "   [skip] Checkpoint not found: ${MODEL_DIR}"
        SKIP=$((SKIP + 1))
        continue
    fi

    REP_PATH="${NSYS_DIR}/${POINT_TAG}"   # nsys adds .nsys-rep automatically

    # Step 1: nsys profile
    # --profile_nsys 会加 NVTX 注解 + cudaProfilerStart/Stop（只采 measure loop）
    # --capture-range cudaProfilerApi 让 nsys 仅捕获 profiler API 范围内的 kernel
    echo "   [profile] → ${REP_PATH}.nsys-rep"
    nsys profile \
        --trace cuda,nvtx \
        --capture-range cudaProfilerApi \
        --output "${REP_PATH}" \
        --force-overwrite true \
        python eval_encoder/scripts/analyze_compute.py \
            --model_dir  "${MODEL_DIR}" \
            --task       "${TASK}" \
            --backend    "${BACKEND}" \
            --input_mode "${INPUT_MODE}" \
            --dtype      "${DTYPE}" \
            --seq_len    "${SEQ_LEN}" \
            --batch_size "${BATCH_SIZE}" \
            --warmup     "${WARMUP}" \
            --measure    "${MEASURE}" \
            --profile_nsys \
    && OK=$((OK + 1)) || { FAIL=$((FAIL + 1)); echo "   [FAILED] nsys profile for ${POINT_TAG}"; continue; }

    # Step 2: extract cuda_gpu_kern_sum, append to summary txt with section header
    echo "   [stats] → ${SUMMARY_TXT}"
    printf "\n════ %s ════\n" "${POINT_TAG}" >> "${SUMMARY_TXT}"
    nsys stats \
        --report cuda_gpu_kern_sum \
        --quiet \
        "${REP_PATH}.nsys-rep" \
    >> "${SUMMARY_TXT}" 2>&1

    echo ""
done

echo "══════════════════════════════════════════════════════════════════════"
echo "  Profiling 完成  成功=${OK}  失败=${FAIL}  跳过=${SKIP}"
echo ""

if [[ "${OK}" -eq 0 ]]; then
    echo "[warn] 没有成功的 profiling 点，跳过 parse + plot"
    exit 0
fi

# ── Step 3: parse summary → CSV ───────────────────────────────────────────────
echo "── 解析 nsys_summary.txt → ${OUT_CSV}"
python eval_encoder/scripts/parse_nsys_summary.py \
    --input   "${SUMMARY_TXT}" \
    --out_csv "${OUT_CSV}"
echo ""

# ── Step 4: 生成图 ────────────────────────────────────────────────────────────
echo "── 生成 kernel analysis 图 → ${FIGURES_DIR}"
python eval_encoder/scripts/plot_nsys_kernel.py \
    --csv    "${OUT_CSV}" \
    --outdir "${FIGURES_DIR}"
echo ""

echo "══════════════════════════════════════════════════════════════════════"
echo "  expD 完成"
echo "  .nsys-rep → ${NSYS_DIR}/"
echo "  summary   → ${SUMMARY_TXT}"
echo "  csv       → ${OUT_CSV}"
echo "  figure    → ${FIGURES_DIR}/nsys_kernel_analysis_${TASK}_${DTYPE}_seq${SEQ_LEN}.png"
echo "  (plot_nsys_kernel.py 固定命名为 nsys_kernel_analysis_mnli_bf16_seq512.png)"
echo "══════════════════════════════════════════════════════════════════════"

[[ "${FAIL}" -eq 0 ]]
