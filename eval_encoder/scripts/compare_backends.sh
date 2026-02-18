#!/bin/bash
################################################################################
# Backend 对比测试脚本
# 自动对比 naive 和 flashsvd 两个后端的性能
################################################################################

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_step() {
    echo -e "${GREEN}==>${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# 检查目录
if [[ ! -f "eval_encoder/scripts/one_click_glue.sh" ]]; then
    echo "Error: Please run from lowrankarena/ directory"
    exit 1
fi

# 配置
METHOD="${METHOD:-fwsvd}"
RETENTION="${RETENTION:-0.5}"
RANK="${RANK:-}"
TASKS="${TASKS:-sst2}"
SKIP_FINETUNING="${SKIP_FINETUNING:-true}"  # 默认跳过微调,快速对比
USE_TASK_MODELS="${USE_TASK_MODELS:-true}"

echo "════════════════════════════════════════════════════════════════"
echo "  Backend 对比测试"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "配置:"
echo "  Method:      $METHOD"
echo "  Retention:   ${RETENTION:-N/A}"
echo "  Rank:        ${RANK:-auto}"
echo "  Tasks:       $TASKS"
echo "  Skip Finetune: $SKIP_FINETUNING"
echo ""
echo "将测试以下后端:"
echo "  1. naive"
echo "  2. flashsvd"
echo ""

# 测试 naive
print_step "Testing backend: naive"
BACKEND=naive \
METHOD=$METHOD \
RETENTION=$RETENTION \
RANK=$RANK \
TASKS=$TASKS \
SKIP_FINETUNING=$SKIP_FINETUNING \
USE_TASK_MODELS=$USE_TASK_MODELS \
bash eval_encoder/scripts/one_click_glue.sh

echo ""
print_info "naive backend test complete. Starting flashsvd..."

# 测试 flashsvd
print_step "Testing backend: flashsvd"
BACKEND=flashsvd \
METHOD=$METHOD \
RETENTION=$RETENTION \
RANK=$RANK \
TASKS=$TASKS \
SKIP_FINETUNING=$SKIP_FINETUNING \
USE_TASK_MODELS=$USE_TASK_MODELS \
bash eval_encoder/scripts/one_click_glue.sh

# 完成
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Backend 对比测试完成!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "结果文件:"
ls -lh eval_encoder/glue_results/glue_results_${METHOD}_*.json | tail -2
echo ""
echo "查看对比结果:"
echo "  python eval_encoder/scripts/generate_comparison_table.py \\"
echo "    eval_encoder/glue_results/glue_results_${METHOD}_*.json"
echo ""
