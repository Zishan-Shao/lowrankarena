#!/bin/bash
################################################################################
# Test FWSVD: Full-matrix mode vs Per-head mode
# 比较 3 个 GLUE 任务上的表现：RTE, MRPC, CoLA
#
# 测试矩阵：
# 1. Dense baseline (naive backend)
# 2. FWSVD full-matrix mode: rank_attn=256, rank_ffn=256, rank_wo=256, qkv_mode=full (naive only)
# 3. FWSVD per-head mode: rank_attn=22, rank_ffn=240, rank_wo=256, qkv_mode=per_head (naive)
# 4. FWSVD per-head mode: rank_attn=22, rank_ffn=240, rank_wo=256, qkv_mode=per_head (flashsvd)
################################################################################

set -e

# 配置
TASKS="rte mrpc cola"
CALIB_BATCHES=16  # 增加校准批次以获得更好的 Fisher 估计

echo "════════════════════════════════════════════════════════════════════"
echo "FWSVD Full vs Per-head Comparison Test"
echo "════════════════════════════════════════════════════════════════════"
echo "Tasks: $TASKS"
echo "Calibration batches: $CALIB_BATCHES"
echo "════════════════════════════════════════════════════════════════════"

# Test 1: Dense baseline
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "Test 1: Dense Baseline (Naive)"
echo "════════════════════════════════════════════════════════════════════"
METHOD=dense \
BACKEND=naive \
TASKS="$TASKS" \
USE_TASK_MODELS=true \
SKIP_FINETUNING=true \
NON_INTERACTIVE=true \
bash eval_encoder/scripts/one_click_glue.sh

# Test 2: FWSVD full-matrix mode (naive only)
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "Test 2: FWSVD Full-Matrix Mode (Naive)"
echo "════════════════════════════════════════════════════════════════════"
echo "Config: rank_attn=256, rank_ffn=256, rank_wo=256, qkv_mode=full"
echo "Note: Full mode applies SVD to entire 768x768 matrices (paper-style)"
echo "════════════════════════════════════════════════════════════════════"
METHOD=fwsvd \
BACKEND=naive \
RANK_ATTN=256 \
RANK_FFN=256 \
RANK_WO=256 \
QKV_MODE=full \
CALIB_BATCHES=$CALIB_BATCHES \
TASKS="$TASKS" \
USE_TASK_MODELS=true \
SKIP_FINETUNING=true \
NON_INTERACTIVE=true \
bash eval_encoder/scripts/one_click_glue.sh

# Test 3: FWSVD per-head mode (naive)
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "Test 3: FWSVD Per-Head Mode (Naive)"
echo "════════════════════════════════════════════════════════════════════"
echo "Config: rank_attn=22, rank_ffn=240, rank_wo=256, qkv_mode=per_head"
echo "Note: Per-head mode applies SVD to each attention head separately (64x64)"
echo "════════════════════════════════════════════════════════════════════"
METHOD=fwsvd \
BACKEND=naive \
RANK_ATTN=22 \
RANK_FFN=240 \
RANK_WO=256 \
QKV_MODE=per_head \
CALIB_BATCHES=$CALIB_BATCHES \
TASKS="$TASKS" \
USE_TASK_MODELS=true \
SKIP_FINETUNING=true \
NON_INTERACTIVE=true \
bash eval_encoder/scripts/one_click_glue.sh

# Test 4: FWSVD per-head mode (flashsvd)
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "Test 4: FWSVD Per-Head Mode (FlashSVD)"
echo "════════════════════════════════════════════════════════════════════"
echo "Config: rank_attn=22, rank_ffn=240, rank_wo=256, qkv_mode=per_head"
echo "Note: FlashSVD backend uses optimized Triton kernels"
echo "════════════════════════════════════════════════════════════════════"
METHOD=fwsvd \
BACKEND=flashsvd \
RANK_ATTN=22 \
RANK_FFN=240 \
RANK_WO=256 \
QKV_MODE=per_head \
CALIB_BATCHES=$CALIB_BATCHES \
TASKS="$TASKS" \
USE_TASK_MODELS=true \
SKIP_FINETUNING=true \
NON_INTERACTIVE=true \
bash eval_encoder/scripts/one_click_glue.sh

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "✅ All tests complete!"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Results summary:"
echo "  - Dense baseline (naive)"
echo "  - FWSVD full-matrix (naive, rank=256 全矩阵分解)"
echo "  - FWSVD per-head (naive, rank=22 每头分解)"
echo "  - FWSVD per-head (flashsvd, rank=22 每头分解)"
echo ""
echo "Results saved to:"
echo "  eval_encoder/glue_results/glue_results_*.json"
echo ""
