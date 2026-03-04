#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# expD.sh  —  Kernel-Level Analysis (nsys + ncu)
#
# NVIDIA Nsight Systems / Nsight Compute profiling on 4 representative points, measuring:
#   nsys phase (time distribution):
#     - GEMM kernel variant count (reflects rank heterogeneity)
#     - Triton fused kernel time share vs cuBLAS GEMM time share
#     - Kernel fragmentation degree (fragmentation_ratio)
#   ncu phase (CTA parallelism / occupancy root cause):
#     - sm__ctas_active.avg              (CTA parallelism)
#     - sm__warps_active occupancy %     (SM occupancy)
#     - launch__occupancy_limit_*        (bottleneck type: registers/shmem/warp)
#     - stall ratio                      (supporting evidence)
#
# 6 profiling points (consistent with plot_nsys_kernel.py POINT_META)
# ────────────────────────────────────────────────────────────────────────────
#   mnli_svd_naive          SVD     + naive
#   mnli_svd_flashsvd       SVD     + flashsvd (v1.0)
#   mnli_svd_flashsvd15     SVD     + flashsvd15 (v1.5)
#   mnli_adasvd_naive       AdaSVD  + naive
#   mnli_adasvd_flashsvd    AdaSVD  + flashsvd (v1.0)
#   mnli_adasvd_flashsvd15  AdaSVD  + flashsvd15 (v1.5)
#
# Dependencies
# ────────────────────────────────────────────────────────────────────────────
#   nsys  (NVIDIA Nsight Systems CLI)
#   ncu   (NVIDIA Nsight Compute CLI)   — required only for ncu phase
#   benchmark/analysis/parse_nsys_summary.py
#   benchmark/analysis/parse_ncu_csv.py
#   benchmark/figures/plot_nsys_kernel.py
#
# Prerequisite: expA must have generated the following checkpoints (missing = skip):
#   compressed_models/bert/svd/mnli/svd_r256_naive/
#   compressed_models/bert/adasvd/adasvd_b0.5_flashsvd/   (sst2, no task subfolder)
#
# Usage
# ────────────────────────────────────────────────────────────────────────────
#   cd lowrankarena/
#   bash benchmark/expD.sh              # run both phases
#   PHASES=nsys bash benchmark/expD.sh  # nsys only
#   PHASES=ncu  bash benchmark/expD.sh  # ncu only
#
# Reproducibility switches (recommended on multi-GPU machines)
# ────────────────────────────────────────────────────────────────────────────
#   GPU_ID=0                        bind to a specific GPU (sets CUDA_VISIBLE_DEVICES)
#   TAG=mnli_bf16_s512_b32          output filename suffix to avoid overwriting different configs
#                                   auto-derived by default: ${TASK}_${DTYPE}_s${SEQ_LEN}_b${BATCH_SIZE}
#
#   Examples:
#   GPU_ID=0 TAG=mnli_bf16_s512_b32 bash benchmark/expD.sh
#   GPU_ID=1 TAG=mnli_bf16_s256_b32 SEQ_LEN=256 bash benchmark/expD.sh
#
# Overridable variables
#   PHASES="nsys ncu"
#   TASK=mnli
#   DTYPE=bf16
#   SEQ_LEN=512
#   BATCH_SIZE=32
#   INPUT_MODE=synthetic
#   WARMUP=5       MEASURE=50        # nsys phase (many steps to sample kernel diversity)
#   NCU_WARMUP=1   NCU_MEASURE=3     # ncu phase (ncu is very slow; few steps are sufficient)
#   NCU_METRICS="sm__ctas_active.avg,sm__warps_active.avg.pct_of_peak_sustained_active,\
#                launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,\
#                launch__occupancy_limit_warps,\
#                smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
#                smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"
#   MODEL_BASE_DIR=compressed_models/bert
#   NSYS_DIR=experiments/nsys
#   SUMMARY_TXT=experiments/nsys/nsys_summary_<TAG>.txt  # includes TAG by default
#   OUT_CSV=experiments/results/expD_<TAG>.csv          # includes TAG by default
#   OUT_NCU_CSV=experiments/results/expD_ncu_<TAG>.csv  # includes TAG by default
#   FIGURES_DIR=experiments/figs
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-experiments/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/expD_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[log] → ${LOG_FILE}"

# ── Configuration ─────────────────────────────────────────────────────────────
PHASES="${PHASES:-nsys ncu}"

TASK="${TASK:-mnli}"
DTYPE="${DTYPE:-bf16}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
INPUT_MODE="${INPUT_MODE:-synthetic}"

# ── Reproducibility switches ──────────────────────────────────────────────────
# GPU_ID: binds CUDA_VISIBLE_DEVICES to ensure same GPU (prevents scheduling to different GPU on multi-GPU machines)
GPU_ID="${GPU_ID:-}"
if [[ -n "${GPU_ID}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

# TAG: filename suffix to distinguish outputs of different configs (prevents overwriting & eases merging)
#   auto-derived: <task>_<dtype>_s<seq_len>_b<batch> (can be overridden manually)
TAG="${TAG:-${TASK}_${DTYPE}_s${SEQ_LEN}_b${BATCH_SIZE}}"

RANK_FFN="${RANK_FFN:-256}"
BUDGET="${BUDGET:-0.5}"
ADASVD_TASK="${ADASVD_TASK:-sst2}"

# nsys phase
WARMUP="${WARMUP:-5}"
MEASURE="${MEASURE:-50}"

# ncu phase (ncu itself is very expensive; 1 warmup + 3 measure is enough to cover all kernel types)
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

MODEL_BASE_DIR="${MODEL_BASE_DIR:-compressed_models/bert}"
NSYS_DIR="${NSYS_DIR:-experiments/nsys}"
SUMMARY_TXT="${SUMMARY_TXT:-${NSYS_DIR}/nsys_summary_${TAG}.txt}"
OUT_CSV="${OUT_CSV:-experiments/results/expD_${TAG}.csv}"
OUT_NCU_CSV="${OUT_NCU_CSV:-experiments/results/expD_ncu_${TAG}.csv}"
FIGURES_DIR="${FIGURES_DIR:-experiments/figs}"

mkdir -p "${NSYS_DIR}" "${FIGURES_DIR}"

# ── Profiling point definitions (consistent with plot_nsys_kernel.py POINT_META) ──────
POINTS=(
    "${TASK}_svd_naive:svd:naive"
    "${TASK}_svd_flashsvd:svd:flashsvd"
    "${TASK}_svd_flashsvd15:svd:flashsvd15"
    "${ADASVD_TASK}_adasvd_naive:adasvd:naive"
    "${ADASVD_TASK}_adasvd_flashsvd:adasvd:flashsvd"
    "${ADASVD_TASK}_adasvd_flashsvd15:adasvd:flashsvd15"
)

# ── Checkpoint path for each method ──────────────────────────────────────────
# Actual layout: compressed_models/bert/{method}/{task}/{method}_r{rank}_naive
#   SVD:    bert/svd/{task}/svd_r{RANK_FFN}_naive    (uniform rank, load naive for all backends)
#   AdaSVD: bert/adasvd/adasvd_b{BUDGET}_flashsvd    (no task subfolder; use ADASVD_TASK for the run)
_model_dir() {
    local method="$1"
    local task_arg="$2"
    if [[ "${method}" == "adasvd" ]]; then
        echo "${MODEL_BASE_DIR}/adasvd/adasvd_b${BUDGET}_flashsvd"
    else
        echo "${MODEL_BASE_DIR}/${method}/${task_arg}/${method}_r${RANK_FFN}_naive"
    fi
}

echo "══════════════════════════════════════════════════════════════════════"
echo "  expD — Kernel-Level Analysis"
echo "  phases:     ${PHASES}"
echo "  task:       ${TASK} (svd)  ${ADASVD_TASK} (adasvd)   dtype: ${DTYPE}   seq_len: ${SEQ_LEN}   bs: ${BATCH_SIZE}"
echo "  input_mode: ${INPUT_MODE}"
echo "  tag:        ${TAG}   gpu: ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "  nsys:       warmup=${WARMUP}  measure=${MEASURE}  → ${OUT_CSV}"
echo "  ncu:        warmup=${NCU_WARMUP}  measure=${NCU_MEASURE}  → ${OUT_NCU_CSV}"
echo "  nsys_dir:   ${NSYS_DIR}"
echo "══════════════════════════════════════════════════════════════════════"

# ── GPU info (written to summary file to provide reproducibility evidence) ───────
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
# Phase nsys: time distribution profiling
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "${PHASES}" == *"nsys"* ]]; then
    echo "══ Phase nsys: Nsight Systems ════════════════════════════════════"

    if ! command -v nsys &>/dev/null; then
        echo "[error] nsys not found. Install NVIDIA Nsight Systems."
        exit 1
    fi

    # Clear (regenerate) summary txt
    > "${SUMMARY_TXT}"

    for ENTRY in "${POINTS[@]}"; do
        POINT_TAG="${ENTRY%%:*}"
        REST="${ENTRY#*:}"
        METHOD="${REST%%:*}"
        BACKEND="${REST#*:}"

        # Determine which task to use (AdaSVD checkpoint is task-specific)
        if [[ "${METHOD}" == "adasvd" ]]; then
            PROFILE_TASK="${ADASVD_TASK}"
        else
            PROFILE_TASK="${TASK}"
        fi

        MODEL_DIR="$(_model_dir "${METHOD}" "${PROFILE_TASK}")"

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
            python benchmark/analysis/analyze_compute.py \
                --model_dir  "${MODEL_DIR}" \
                --task       "${PROFILE_TASK}" \
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
            --force-export=true \
            --quiet \
            "${REP_PATH}.nsys-rep" \
        >> "${SUMMARY_TXT}" 2>&1 || { FAIL=$((FAIL + 1)); echo "   [FAILED] nsys stats for ${POINT_TAG}"; continue; }

        OK=$((OK + 1))

        echo ""
    done

    echo "  nsys done  success=${OK}  failed=${FAIL}  skipped=${SKIP}"
    echo ""

    if [[ "${OK}" -gt 0 ]]; then
        echo "── Parse nsys_summary.txt → ${OUT_CSV}"
        python benchmark/analysis/parse_nsys_summary.py \
            --input   "${SUMMARY_TXT}" \
            --out_csv "${OUT_CSV}"
        echo ""

        echo "── Generate kernel analysis figure → ${FIGURES_DIR}"
        python benchmark/figures/plot_nsys_kernel.py \
            --csv    "${OUT_CSV}" \
            --outdir "${FIGURES_DIR}"
        echo ""
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase ncu: CTA parallelism / Occupancy root cause
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

            if [[ "${METHOD}" == "adasvd" ]]; then
                PROFILE_TASK="${ADASVD_TASK}"
            else
                PROFILE_TASK="${TASK}"
            fi

            MODEL_DIR="$(_model_dir "${METHOD}" "${PROFILE_TASK}")"

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
                python benchmark/analysis/analyze_compute.py \
                    --model_dir  "${MODEL_DIR}" \
                    --task       "${PROFILE_TASK}" \
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

        echo "  ncu done  success=${NCU_OK}  failed=${NCU_FAIL}  skipped=${NCU_SKIP}"
        echo ""

        if [[ "${NCU_OK}" -gt 0 ]]; then
            echo "── Parse ncu_raw_*.csv → ${OUT_NCU_CSV}"
            python benchmark/analysis/parse_ncu_csv.py \
                --ncu_dir "${NSYS_DIR}" \
                --task    "${TASK}" \
                --out_csv "${OUT_NCU_CSV}"
            echo ""
        fi
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
echo "  expD done  success=${OK}  failed=${FAIL}  skipped=${SKIP}"
echo ""
echo "  Output:"
[[ "${PHASES}" == *"nsys"* ]] && echo "    nsys csv    → ${OUT_CSV}"
[[ "${PHASES}" == *"nsys"* ]] && echo "    nsys figure → ${FIGURES_DIR}/nsys_kernel_analysis_${TASK}_${DTYPE}_seq${SEQ_LEN}.png"
[[ "${PHASES}" == *"ncu"*  ]] && echo "    ncu csv     → ${OUT_NCU_CSV}"
[[ "${PHASES}" == *"ncu"*  ]] && echo "    ncu raw     → ${NSYS_DIR}/ncu_raw_*.csv"
echo "══════════════════════════════════════════════════════════════════════"

[[ "${FAIL}" -eq 0 ]]
