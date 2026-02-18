#!/bin/bash
################################################################################
# eval_encoder 清理脚本
#
# 用法:
#   bash eval_encoder/cleanup.sh conservative  # 保守清理 (~200MB)
#   bash eval_encoder/cleanup.sh standard      # 标准清理 (~500MB)
#   bash eval_encoder/cleanup.sh deep          # 深度清理 (~1.9GB)
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

print_info() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# 检查参数
CLEANUP_MODE="${1:-conservative}"

if [[ ! "$CLEANUP_MODE" =~ ^(conservative|standard|deep)$ ]]; then
    print_error "Invalid cleanup mode: $CLEANUP_MODE"
    echo ""
    echo "Usage: bash eval_encoder/cleanup.sh [conservative|standard|deep]"
    echo ""
    echo "Modes:"
    echo "  conservative - 仅删除缓存和归档文件 (~200MB)"
    echo "  standard     - 删除缓存、归档、日志 (~500MB)"
    echo "  deep         - 删除所有临时文件和模型 (~1.9GB)"
    exit 1
fi

print_header "eval_encoder 清理脚本 - 模式: $CLEANUP_MODE"

# 显示当前占用
echo "当前目录占用:"
du -sh . 2>/dev/null || echo "无法计算大小"
echo ""

# ══════════════════════════════════════════════════════════════════════════
# 函数定义
# ══════════════════════════════════════════════════════════════════════════

safe_remove() {
    local path="$1"
    if [ -e "$path" ]; then
        rm -rf "$path"
        print_info "已删除: $path"
        return 0
    else
        return 1
    fi
}

# ══════════════════════════════════════════════════════════════════════════
# 保守清理 (Conservative)
# ══════════════════════════════════════════════════════════════════════════

cleanup_conservative() {
    print_header "执行保守清理"

    echo "清理 Python 缓存..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -exec rm -f {} + 2>/dev/null || true
    print_info "Python 缓存已清理"

    echo ""
    echo "清理临时输出..."
    safe_remove "ars_out"

    echo ""
    echo "清理归档目录..."
    safe_remove "eval_results/archived_csvs"
    safe_remove "eval_results/archived_adasvd_outputs"
    safe_remove "docs/archived"
    safe_remove "logs/archived"
    safe_remove "scripts/archived"

    echo ""
    echo "清理备份和测试文件..."
    find eval_results -type f -name "*backup*.csv" -delete 2>/dev/null || true
    find eval_results -type f -name "test_*.csv" -delete 2>/dev/null || true
    safe_remove "eval_results/encoder_runs_flashsvd_comparison.csv"
    safe_remove "eval_results/encoder_runs_flashsvd_longseq_test.csv"
    safe_remove "eval_results/encoder_runs_small_ranks_comparison.csv"
    safe_remove "eval_results/encoder_runs_sst2_adasvd_refactored_5budgets.csv"
    safe_remove "eval_results/complete_e2e_retest.csv"
    safe_remove "eval_results/comprehensive_test_results.csv"
    print_info "备份文件已清理"

    echo ""
    print_info "保守清理完成！"
}

# ══════════════════════════════════════════════════════════════════════════
# 标准清理 (Standard)
# ══════════════════════════════════════════════════════════════════════════

cleanup_standard() {
    cleanup_conservative

    echo ""
    print_header "执行标准清理 (额外清理)"

    echo "清理训练日志..."
    # 保留最新的3个日志文件
    if [ -d "logs" ]; then
        cd logs
        ls -t *.log 2>/dev/null | tail -n +4 | xargs -r rm -f
        cd ..
        print_info "旧日志已清理 (保留最新3个)"
    fi

    echo ""
    echo "清理旧的CSV结果..."
    if [ -d "eval_results" ]; then
        cd eval_results
        # 保留 encoder_runs.csv 和 final/ 目录
        ls -t *.csv 2>/dev/null | grep -v "encoder_runs.csv" | tail -n +3 | xargs -r rm -f
        cd ..
        print_info "旧CSV已清理 (保留最新的和encoder_runs.csv)"
    fi

    echo ""
    print_info "标准清理完成！"
}

# ══════════════════════════════════════════════════════════════════════════
# 深度清理 (Deep)
# ══════════════════════════════════════════════════════════════════════════

cleanup_deep() {
    cleanup_standard

    echo ""
    print_header "执行深度清理 (额外清理)"

    print_warning "即将删除所有模型和日志！"
    echo ""
    echo "将删除:"
    echo "  - models/ (1.5GB)"
    echo "  - finetuned_models/ (419MB)"
    echo "  - logs/ (17MB)"
    echo ""
    read -p "确认删除? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        print_warning "已取消深度清理"
        return
    fi

    echo ""
    echo "清理所有模型..."
    safe_remove "models"
    safe_remove "finetuned_models"

    echo ""
    echo "清理所有日志..."
    safe_remove "logs"
    mkdir -p logs  # 重新创建空目录

    echo ""
    print_info "深度清理完成！"
}

# ══════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════

case "$CLEANUP_MODE" in
    conservative)
        cleanup_conservative
        ;;
    standard)
        cleanup_standard
        ;;
    deep)
        cleanup_deep
        ;;
esac

echo ""
print_header "清理完成"

echo "清理后目录占用:"
du -sh . 2>/dev/null || echo "无法计算大小"
echo ""

echo "可以通过以下命令查看详细变化:"
echo "  git status"
echo ""
