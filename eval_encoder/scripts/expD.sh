#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# expD.sh  —  Kernel-Level Analysis (nsys + ncu)
#
# 对 4 个代表点做 NVIDIA Nsight Systems / Nsight Compute profiling，量化：
#   nsys 阶段（时间分布）：
#     - GEMM kernel 变体数（反映 rank 异质性）
#     - Triton fused kernel 占比 vs cuBLAS GEMM 占比
#     - kernel 碎片化程度（fragmentation_ratio）
#   ncu 阶段（CTA 并行度 / occupancy 根因）：
#     - sm__ctas_active.avg              （CTA 并行度）
#     - sm__warps_active occupancy %     （SM occupancy）
#     - launch__occupancy_limit_*        （瓶颈类型：寄存器/shmem/warp）
#     - stall 比例                       （佐证）
#
# 4 个 profiling 点（与 plot_nsys_kernel.py POINT_META 一致）
# ────────────────────────────────────────────────────────────────────────────
#   mnli_svd_naive          SVD     + naive
#   mnli_svd_flashsvd15     SVD     + flashsvd15
#   mnli_adasvd_naive       AdaSVD  + naive
#   mnli_adasvd_flashsvd15  AdaSVD  + flashsvd15
#
# 依赖
# ────────────────────────────────────────────────────────────────────────────
#   nsys  (NVIDIA Nsight Systems CLI)
#   ncu   (NVIDIA Nsight Compute CLI)   — 仅 ncu 阶段需要
#   eval_encoder/scripts/parse_nsys_summary.py
#   eval_encoder/scripts/parse_ncu_csv.py
#   eval_encoder/scripts/plot_nsys_kernel.py
#
# ⚠️ 前提：expA 已生成下列 checkpoint（缺一则对应点 skip）：
#   eval_encoder/models/mnli/svd_ra48_rf256_rw208_per_head_naive/
#   eval_encoder/models/mnli/adasvd_b0.527_per_head_naive/
#
# 用法
# ────────────────────────────────────────────────────────────────────────────
#   cd lowrankarena/
#   bash eval_encoder/scripts/expD.sh              # 两个阶段都跑
#   PHASES=nsys bash eval_encoder/scripts/expD.sh  # 只跑 nsys
#   PHASES=ncu  bash eval_encoder/scripts/expD.sh  # 只跑 ncu
#
# 可复现性开关（推荐多卡机器必用）
# ────────────────────────────────────────────────────────────────────────────
#   GPU_ID=0                        绑定到指定 GPU（设置 CUDA_VISIBLE_DEVICES）
#   TAG=mnli_bf16_s512_b32          输出文件名后缀，避免不同配置覆盖彼此
#                                   默认自动派生：${TASK}_${DTYPE}_s${SEQ_LEN}_b${BATCH_SIZE}
#
#   示例：
#   GPU_ID=0 TAG=mnli_bf16_s512_b32 bash eval_encoder/scripts/expD.sh
#   GPU_ID=1 TAG=mnli_bf16_s256_b32 SEQ_LEN=256 bash eval_encoder/scripts/expD.sh
#
# 可覆盖变量
#   PHASES="nsys ncu"
#   TASK=mnli
#   DTYPE=bf16
#   SEQ_LEN=512
#   BATCH_SIZE=32
#   INPUT_MODE=synthetic
#   WARMUP=5       MEASURE=50        # nsys 阶段（多步采样 kernel 多样性）
#   NCU_WARMUP=1   NCU_MEASURE=3     # ncu 阶段（ncu 本身很慢，少步即可）
#   NCU_METRICS="sm__ctas_active.avg,sm__warps_active.avg.pct_of_peak_sustained_active,\
#                launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,\
#                launch__occupancy_limit_warps,\
#                smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
#                smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"
#   MODEL_BASE_DIR=eval_encoder/models
#   NSYS_DIR=eval_encoder/eval_results/nsys
#   SUMMARY_TXT=eval_encoder/eval_results/nsys/nsys_summary_<TAG>.txt  # 默认含 TAG
#   OUT_CSV=eval_encoder/eval_results/expD_<TAG>.csv          # 默认含 TAG
#   OUT_NCU_CSV=eval_encoder/eval_results/expD_ncu_<TAG>.csv  # 默认含 TAG
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
PHASES="${PHASES:-nsys ncu}"

TASK="${TASK:-mnli}"
DTYPE="${DTYPE:-bf16}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
INPUT_MODE="${INPUT_MODE:-synthetic}"

# ── 可复现性开关 ───────────────────────────────────────────────────────────────
# GPU_ID: 绑定 CUDA_VISIBLE_DEVICES，确保同一卡（多卡机器防止调度到不同 GPU）
GPU_ID="${GPU_ID:-}"
if [[ -n "${GPU_ID}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

# TAG: 文件名后缀，区分不同配置的输出（防覆盖 & 方便 merge）
#   自动派生：<task>_<dtype>_s<seq_len>_b<batch>（可手动覆盖）
TAG="${TAG:-${TASK}_${DTYPE}_s${SEQ_LEN}_b${BATCH_SIZE}}"

RANK_ATTN="${RANK_ATTN:-48}"
RANK_FFN="${RANK_FFN:-256}"
RANK_WO="${RANK_WO:-208}"
QKV_MODE="${QKV_MODE:-per_head}"
BUDGET="${BUDGET:-0.527}"

# nsys 阶段
WARMUP="${WARMUP:-5}"
MEASURE="${MEASURE:-50}"

# ncu 阶段（ncu 本身开销极高，1 warmup + 3 measure 足够覆盖所有 kernel 类型）
NCU_WARMUP="${NCU_WARMUP:-1}"
NCU_MEASURE="${NCU_MEASURE:-3}"
_DEFAULT_NCU_METRICS="sm__ctas_active.avg"
_DEFAULT_NCU_METRICS+=",sm__warps_active.avg.pct_of_peak_sustained_active"
_DEFAULT_NCU_METRICS+=",launch__occupancy_limit_registers"
_DEFAULT_NCU_METRICS+=",launch__occupancy_limit_shared_mem"
_DEFAULT_NCU_METRICS+=",launch__occupancy_limit_warps"
_DEFAULT_NCU_METRICS+=",smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
_DEFAULT_NCU_METRICS+=",smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"
NCU_METRICS="${NCU_METRICS:-${_DEFAULT_NCU_METRICS}}"

MODEL_BASE_DIR="${MODEL_BASE_DIR:-eval_encoder/models}"
NSYS_DIR="${NSYS_DIR:-eval_encoder/eval_results/nsys}"
SUMMARY_TXT="${SUMMARY_TXT:-${NSYS_DIR}/nsys_summary_${TAG}.txt}"
OUT_CSV="${OUT_CSV:-eval_encoder/eval_results/expD_${TAG}.csv}"
OUT_NCU_CSV="${OUT_NCU_CSV:-eval_encoder/eval_results/expD_ncu_${TAG}.csv}"
FIGURES_DIR="${FIGURES_DIR:-eval_encoder/eval_results/figures}"

mkdir -p "${NSYS_DIR}" "${FIGURES_DIR}"

# ── profiling 点定义（与 plot_nsys_kernel.py POINT_META 一致）─────────────────
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
echo "  expD — Kernel-Level Analysis"
echo "  phases:     ${PHASES}"
echo "  task:       ${TASK}   dtype: ${DTYPE}   seq_len: ${SEQ_LEN}   bs: ${BATCH_SIZE}"
echo "  input_mode: ${INPUT_MODE}"
echo "  tag:        ${TAG}   gpu: ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "  nsys:       warmup=${WARMUP}  measure=${MEASURE}  → ${OUT_CSV}"
echo "  ncu:        warmup=${NCU_WARMUP}  measure=${NCU_MEASURE}  → ${OUT_NCU_CSV}"
echo "  nsys_dir:   ${NSYS_DIR}"
echo "══════════════════════════════════════════════════════════════════════"

# ── GPU 信息（写入 summary 文件，提供可复现性证据）─────────────────────────────
GPU_INFO_FILE="${NSYS_DIR}/gpu_info_${TAG}.txt"
mkdir -p "${NSYS_DIR}"
{
    echo "# expD GPU info  tag=${TAG}  date=$(date -Iseconds)"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap \
               --format=csv 2>/dev/null || echo "(nvidia-smi unavailable)"
    echo ""
    nvcc --version 2>/dev/null | grep -E "release|V[0-9]" || echo "(nvcc unavailable)"
    python - <<'PYEOF'
import torch
print(f"torch={torch.__version__}  cuda={torch.version.cuda}  cudnn={torch.backends.cudnn.version()}")
PYEOF
} | tee "${GPU_INFO_FILE}"
echo ""
echo ""

OK=0
FAIL=0
SKIP=0

# ═══════════════════════════════════════════════════════════════════════════════
# Phase nsys: 时间分布 profiling
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "${PHASES}" == *"nsys"* ]]; then
    echo "══ Phase nsys: Nsight Systems ════════════════════════════════════"

    if ! command -v nsys &>/dev/null; then
        echo "[error] nsys not found. Install NVIDIA Nsight Systems."
        exit 1
    fi

    # 清空（重新生成）summary txt
    > "${SUMMARY_TXT}"

    for ENTRY in "${POINTS[@]}"; do
        POINT_TAG="${ENTRY%%:*}"
        REST="${ENTRY#*:}"
        METHOD="${REST%%:*}"
        BACKEND="${REST#*:}"

        SUBDIR="$(_model_subdir "${METHOD}")"
        MODEL_DIR="${MODEL_BASE_DIR}/${TASK}/${SUBDIR}"

        echo "── ${POINT_TAG}  (${MODEL_DIR})"

        if [[ ! -d "${MODEL_DIR}" ]]; then
            echo "   [skip] Checkpoint not found: ${MODEL_DIR}"
            SKIP=$((SKIP + 1))
            continue
        fi

        REP_PATH="${NSYS_DIR}/${POINT_TAG}"

        echo "   [profile] → ${REP_PATH}.nsys-rep"
        rc=0
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
        || rc=$?

        # exit=143 (SIGTERM) is normal: nsys sends SIGTERM to the target process
        # after --capture-range cudaProfilerApi ends.  Treat it as success if
        # the .nsys-rep file was actually generated.
        if [[ $rc -ne 0 && $rc -ne 143 ]]; then
            FAIL=$((FAIL + 1)); echo "   [FAILED] nsys profile for ${POINT_TAG} (exit=${rc})"; continue
        fi
        if [[ ! -f "${REP_PATH}.nsys-rep" ]]; then
            FAIL=$((FAIL + 1)); echo "   [FAILED] nsys profile for ${POINT_TAG}: .nsys-rep not generated"; continue
        fi

        echo "   [stats] → ${SUMMARY_TXT}"
        printf "\n════ %s ════\n" "${POINT_TAG}" >> "${SUMMARY_TXT}"
        nsys stats \
            --report cuda_gpu_kern_sum \
            --quiet \
            "${REP_PATH}.nsys-rep" \
        >> "${SUMMARY_TXT}" 2>&1 || {
            echo "   [FAILED] nsys stats for ${POINT_TAG} — stderr:"
            nsys stats --report cuda_gpu_kern_sum --quiet "${REP_PATH}.nsys-rep" 2>&1 | head -5 || true
            FAIL=$((FAIL + 1)); continue
        }

        OK=$((OK + 1))

        echo ""
    done

    echo "  nsys 完成  成功=${OK}  失败=${FAIL}  跳过=${SKIP}"
    echo ""

    if [[ "${OK}" -gt 0 ]]; then
        echo "── 解析 nsys_summary.txt → ${OUT_CSV}"
        python eval_encoder/scripts/parse_nsys_summary.py \
            --input   "${SUMMARY_TXT}" \
            --out_csv "${OUT_CSV}"
        echo ""

        echo "── 生成 kernel analysis 图 → ${FIGURES_DIR}"
        python eval_encoder/scripts/plot_nsys_kernel.py \
            --csv    "${OUT_CSV}" \
            --outdir "${FIGURES_DIR}"
        echo ""
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase ncu: CTA 并行度 / Occupancy 根因
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "${PHASES}" == *"ncu"* ]]; then
    echo "══ Phase ncu: Nsight Compute (occupancy / CTA) ═══════════════════"
    echo "   metrics: ${NCU_METRICS}"
    echo "   warmup:  ${NCU_WARMUP}   measure: ${NCU_MEASURE}"
    echo "   out_csv: ${OUT_NCU_CSV}"
    echo ""

    if ! command -v ncu &>/dev/null; then
        echo "[skip] ncu not found — skipping ncu phase."
        echo "       Install NVIDIA Nsight Compute: https://developer.nvidia.com/nsight-compute"
        SKIP=$((SKIP + 1))
    else
        NCU_OK=0
        NCU_FAIL=0
        NCU_SKIP=0

        for ENTRY in "${POINTS[@]}"; do
            POINT_TAG="${ENTRY%%:*}"
            REST="${ENTRY#*:}"
            METHOD="${REST%%:*}"
            BACKEND="${REST#*:}"

            SUBDIR="$(_model_subdir "${METHOD}")"
            MODEL_DIR="${MODEL_BASE_DIR}/${TASK}/${SUBDIR}"

            echo "── ${POINT_TAG}  (${MODEL_DIR})"

            if [[ ! -d "${MODEL_DIR}" ]]; then
                echo "   [skip] Checkpoint not found: ${MODEL_DIR}"
                NCU_SKIP=$((NCU_SKIP + 1))
                continue
            fi

            NCU_REP="${NSYS_DIR}/ncu_${POINT_TAG}"
            NCU_RAW="${NSYS_DIR}/ncu_raw_${POINT_TAG}.csv"
            NCU_LOG="${NSYS_DIR}/ncu_log_${POINT_TAG}.txt"
            echo "   [ncu step1] profile → ${NCU_REP}.ncu-rep"

            # Step 1: profile → .ncu-rep
            # Python script's stdout/stderr go to NCU_LOG (not polluting the CSV).
            # --capture-range cudaProfilerApi: only captures the measure loop.
            # --target-processes all: follow child python processes.
            rc=0
            ncu \
                --metrics "${NCU_METRICS}" \
                --capture-range cudaProfilerApi \
                --target-processes all \
                --output "${NCU_REP}" \
                --force-overwrite \
                python eval_encoder/scripts/analyze_compute.py \
                    --model_dir  "${MODEL_DIR}" \
                    --task       "${TASK}" \
                    --backend    "${BACKEND}" \
                    --input_mode "${INPUT_MODE}" \
                    --dtype      "${DTYPE}" \
                    --seq_len    "${SEQ_LEN}" \
                    --batch_size "${BATCH_SIZE}" \
                    --warmup     "${NCU_WARMUP}" \
                    --measure    "${NCU_MEASURE}" \
                    --profile_nsys \
            > "${NCU_LOG}" 2>&1 || rc=$?

            if [[ $rc -ne 0 ]]; then
                NCU_FAIL=$((NCU_FAIL + 1))
                echo "   [FAILED] ncu profile for ${POINT_TAG} (exit=${rc})"
                echo "            log → ${NCU_LOG}"
                continue
            fi

            # Step 2: export CSV cleanly from .ncu-rep (no Python stdout mixing)
            echo "   [ncu step2] export CSV → ${NCU_RAW}"
            ncu --import "${NCU_REP}.ncu-rep" --csv --page raw \
            > "${NCU_RAW}" 2>>"${NCU_LOG}" || rc=$?

            if [[ $rc -eq 0 ]]; then
                NCU_OK=$((NCU_OK + 1))
                echo "   [ok] ${NCU_RAW}"
            else
                NCU_FAIL=$((NCU_FAIL + 1))
                echo "   [FAILED] ncu export for ${POINT_TAG} (exit=${rc})"
            fi
            echo ""
        done

        OK=$((OK + NCU_OK))
        FAIL=$((FAIL + NCU_FAIL))
        SKIP=$((SKIP + NCU_SKIP))

        echo "  ncu 完成  成功=${NCU_OK}  失败=${NCU_FAIL}  跳过=${NCU_SKIP}"
        echo ""

        if [[ "${NCU_OK}" -gt 0 ]]; then
            echo "── 解析 ncu_raw_*.csv → ${OUT_NCU_CSV}"
            python eval_encoder/scripts/parse_ncu_csv.py \
                --ncu_dir "${NSYS_DIR}" \
                --task    "${TASK}" \
                --out_csv "${OUT_NCU_CSV}"
            echo ""
        fi
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
echo "  expD 完成  成功=${OK}  失败=${FAIL}  跳过=${SKIP}"
echo ""
echo "  输出："
[[ "${PHASES}" == *"nsys"* ]] && echo "    nsys csv    → ${OUT_CSV}"
[[ "${PHASES}" == *"nsys"* ]] && echo "    nsys figure → ${FIGURES_DIR}/nsys_kernel_analysis_${TASK}_${DTYPE}_seq${SEQ_LEN}.png"
[[ "${PHASES}" == *"ncu"*  ]] && echo "    ncu csv     → ${OUT_NCU_CSV}"
[[ "${PHASES}" == *"ncu"*  ]] && echo "    ncu raw     → ${NSYS_DIR}/ncu_raw_*.csv"
echo "══════════════════════════════════════════════════════════════════════"

[[ "${FAIL}" -eq 0 ]]
