#!/bin/bash
################################################################################
# Backend 对比测试脚本
# 自动对比 naive、flashsvd、flashsvd15 三个后端的性能
# flashsvd15: v1.5 rank-space kernel，支持 bf16/fp16 原生，零 cast overhead
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
DTYPE="${DTYPE:-fp32}"  # flashsvd15 + bf16 = 零 cast overhead

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
echo "  Dtype:       $DTYPE"
echo ""
echo "将测试以下后端:"
echo "  1. naive"
echo "  2. flashsvd"
echo "  3. flashsvd15  (v1.5 rank-space kernel, native fp16/bf16)"
echo ""

# 测试 naive
print_step "Testing backend: naive"
BACKEND=naive \
DTYPE=$DTYPE \
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
DTYPE=$DTYPE \
METHOD=$METHOD \
RETENTION=$RETENTION \
RANK=$RANK \
TASKS=$TASKS \
SKIP_FINETUNING=$SKIP_FINETUNING \
USE_TASK_MODELS=$USE_TASK_MODELS \
bash eval_encoder/scripts/one_click_glue.sh

echo ""
print_info "flashsvd backend test complete. Starting flashsvd15..."

# 测试 flashsvd15
print_step "Testing backend: flashsvd15"
BACKEND=flashsvd15 \
DTYPE=$DTYPE \
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
ls -lh eval_encoder/glue_results/glue_results_${METHOD}_*.json | tail -3
echo ""
echo "查看对比结果:"
echo "  python eval_encoder/scripts/generate_comparison_table.py \\"
echo "    eval_encoder/glue_results/glue_results_${METHOD}_*.json"
echo ""
