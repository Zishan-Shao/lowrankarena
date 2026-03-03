#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# expA.sh  —  Quality Experiment (canonical paper reproduction entrypoint)
#
# 固定 canonical config，一键复现所有质量实验：
#   Phase 1: GLUE (8 tasks, 4 methods, stage1 + stage2)
#   Phase 2: SuperGLUE + HANS + ANLI (7 tasks, 4 methods, naive backend)
#
# Canonical config（论文结果的唯一权威参数集）
# ────────────────────────────────────────────────────────────────────────────
#   QKV_MODE  = per_head
#   RANK_ATTN = 48   RANK_FFN = 256   RANK_WO = 208
#   BUDGET    = 0.527   (AdaSVD，参数量与上面 rank 等价)
#   DTYPE     = bf16
#   SEQ_LEN   = 512    BATCH_SIZE = 32
#
# 用法
# ────────────────────────────────────────────────────────────────────────────
#   cd lowrankarena/
#   bash eval_encoder/scripts/expA.sh                # 全量 (GLUE + SuperGLUE)
#   PHASES=glue bash eval_encoder/scripts/expA.sh    # 只跑 GLUE
#   PHASES=superglue bash eval_encoder/scripts/expA.sh  # 只跑 SuperGLUE
#
# 可覆盖变量
#   PHASES="glue superglue"           # 阶段子集
#   METHODS="svd fwsvd"               # 方法子集（默认 4 个）
#   TASKS_GLUE="mnli stsb"            # GLUE 任务子集
#   TASKS_SUPERGLUE="boolq hans"      # SuperGLUE 任务子集
#   TWO_STAGE=false                   # 只跑 stage1（no_finetune），跳过微调
#   RECOMPRESS=true                   # 强制重新压缩（忽略已有 checkpoint）
#   OUT_CSV=path/to.csv               # 所有 CSV 结果写入路径（GLUE + SuperGLUE 共用）
#
# 输出
#   GLUE JSON : eval_encoder/glue_results/glue_results_{method}_*.json
#               → 通过 eval_results/collect_glue_results.py 汇总为 CSV
#   ALL CSV   : eval_encoder/eval_results/expA.csv（GLUE + SuperGLUE 共用）
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ── 日志 ──────────────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-eval_encoder/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/expA_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[log] → ${LOG_FILE}"

# ── Canonical config（per_head 为论文标准；可通过环境变量覆盖跑 full-matrix）──
#   per_head (默认):  QKV_MODE=per_head  RANK_ATTN=48   param_ratio≈0.527
#   full-matrix:      QKV_MODE=full       RANK_ATTN=312  param_ratio≈0.527
export QKV_MODE="${QKV_MODE:-per_head}"
if [[ -z "${RANK_ATTN:-}" ]]; then
    [[ "${QKV_MODE}" == "full" ]] && export RANK_ATTN=312 || export RANK_ATTN=48
else
    export RANK_ATTN="${RANK_ATTN}"
fi
export RANK_FFN="${RANK_FFN:-256}"
export RANK_WO="${RANK_WO:-208}"
export BUDGET="${BUDGET:-0.527}"
export DTYPE="${DTYPE:-bf16}"
export SEQ_LEN="${SEQ_LEN:-512}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export CALIB_BATCHES="${CALIB_BATCHES:-16}"
export ADASVD_CALIB_SAMPLES="${ADASVD_CALIB_SAMPLES:-4000}"
export ADASVD_STEPS="${ADASVD_STEPS:-800}"

# ── 可覆盖配置 ─────────────────────────────────────────────────────────────────
PHASES="${PHASES:-glue superglue}"
METHODS="${METHODS:-svd fwsvd drone adasvd}"
TASKS_GLUE="${TASKS_GLUE:-cola sst2 mrpc qqp mnli qnli rte stsb}"
TASKS_SUPERGLUE="${TASKS_SUPERGLUE:-boolq rte_sg wic hans anli_r1 anli_r2 anli_r3}"
TWO_STAGE="${TWO_STAGE:-true}"
RECOMPRESS="${RECOMPRESS:-false}"
MODEL_BASE_DIR="${MODEL_BASE_DIR:-eval_encoder/models}"
OUT_CSV="${OUT_CSV:-eval_encoder/eval_results/expA.csv}"

echo "══════════════════════════════════════════════════════════════════════"
echo "  expA — Quality Experiment"
echo "  phases:    ${PHASES}"
echo "  methods:   ${METHODS}"
echo "  config:    qkv=${QKV_MODE} ra${RANK_ATTN}_rf${RANK_FFN}_rw${RANK_WO}  budget=${BUDGET}"
echo "  dtype:     ${DTYPE}   seq_len=${SEQ_LEN}   bs=${BATCH_SIZE}"
echo "  two_stage: ${TWO_STAGE}   recompress: ${RECOMPRESS}"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

OK=0
FAIL=0

# ── Phase 1: GLUE ─────────────────────────────────────────────────────────────
if [[ "${PHASES}" == *"glue"* ]]; then
    echo "══ Phase 1: GLUE ══════════════════════════════════════════════════"
    echo "   tasks:  ${TASKS_GLUE}"
    echo "   stages: $([ "${TWO_STAGE}" = "true" ] && echo "no_finetune + with_finetune" || echo "no_finetune only")"
    echo ""

    TASKS="${TASKS_GLUE}" \
    METHODS="${METHODS}" \
    TWO_STAGE="${TWO_STAGE}" \
    BACKENDS="naive" \
    USE_TASK_MODELS="true" \
    TASK_MODEL_PREFIX="textattack" \
    PRETRAIN_BEFORE_COMPRESS="false" \
    AUTO_FIGURES="false" \
    PERF_CSV="${OUT_CSV}" \
    bash eval_encoder/scripts/compare_all_methods.sh \
    && OK=$((OK + 1)) || FAIL=$((FAIL + 1))

    echo ""
fi

# ── Phase 2: SuperGLUE + HANS + ANLI ─────────────────────────────────────────
if [[ "${PHASES}" == *"superglue"* ]]; then
    echo "══ Phase 2: SuperGLUE / HANS / ANLI ══════════════════════════════"
    echo "   tasks: ${TASKS_SUPERGLUE}"
    echo ""

    TASKS="${TASKS_SUPERGLUE}" \
    METHODS="${METHODS}" \
    BACKENDS="naive" \
    RECOMPRESS="${RECOMPRESS}" \
    MODEL_BASE_DIR="${MODEL_BASE_DIR}" \
    OUT_CSV="${OUT_CSV}" \
    bash eval_encoder/scripts/run_superglue_benchmark.sh \
    && OK=$((OK + 1)) || FAIL=$((FAIL + 1))

    echo ""
fi

echo "══════════════════════════════════════════════════════════════════════"
echo "  expA 完成  成功=${OK}  失败=${FAIL}"
echo ""
echo "  输出："
echo "    GLUE JSON → eval_encoder/glue_results/glue_results_*.json"
echo "    ALL CSV   → ${OUT_CSV}"
echo ""
echo "  汇总图表："
echo "    python eval_encoder/eval_results/collect_glue_results.py"
echo "    python eval_encoder/eval_results/gen_figures.py"
echo "══════════════════════════════════════════════════════════════════════"

[[ "${FAIL}" -eq 0 ]]
