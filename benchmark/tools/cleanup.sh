#!/bin/bash
################################################################################
# Cleanup Script
#
# Usage:
#   bash benchmark/tools/cleanup.sh conservative  # Light cleanup (~200MB)
#   bash benchmark/tools/cleanup.sh standard      # Standard cleanup (~500MB)
#   bash benchmark/tools/cleanup.sh deep          # Full cleanup (~1.9GB)
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

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

# Validate argument
CLEANUP_MODE="${1:-conservative}"

if [[ ! "$CLEANUP_MODE" =~ ^(conservative|standard|deep)$ ]]; then
    print_error "Invalid cleanup mode: $CLEANUP_MODE"
    echo ""
    echo "Usage: bash benchmark/tools/cleanup.sh [conservative|standard|deep]"
    echo ""
    echo "Modes:"
    echo "  conservative - Remove caches and archived files (~200MB)"
    echo "  standard     - Remove caches, archives, and logs (~500MB)"
    echo "  deep         - Remove all temporary files and models (~1.9GB)"
    exit 1
fi

print_header "Cleanup script - mode: $CLEANUP_MODE"

# Show current disk usage
echo "Current directory size:"
du -sh . 2>/dev/null || echo "Unable to calculate size"
echo ""

# ══════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════

safe_remove() {
    local path="$1"
    if [ -e "$path" ]; then
        rm -rf "$path"
        print_info "Removed: $path"
        return 0
    else
        return 1
    fi
}

# ══════════════════════════════════════════════════════════════════════════
# Conservative cleanup
# ══════════════════════════════════════════════════════════════════════════

cleanup_conservative() {
    print_header "Conservative cleanup"

    echo "Removing Python caches..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -exec rm -f {} + 2>/dev/null || true
    print_info "Python caches removed"

    echo ""
    echo "Removing temporary outputs..."
    safe_remove "experiments/ars_out"

    echo ""
    echo "Removing archived directories..."
    safe_remove "experiments/archived_csvs"
    safe_remove "experiments/archived_adasvd_outputs"
    safe_remove "docs/archived"
    safe_remove "experiments/logs/archived"
    safe_remove "benchmark/_retired"

    echo ""
    echo "Removing backup and test files..."
    find experiments -type f -name "*backup*.csv" -delete 2>/dev/null || true
    find experiments -type f -name "test_*.csv" -delete 2>/dev/null || true
    safe_remove "experiments/encoder_runs_flashsvd_comparison.csv"
    safe_remove "experiments/encoder_runs_flashsvd_longseq_test.csv"
    safe_remove "experiments/encoder_runs_small_ranks_comparison.csv"
    safe_remove "experiments/encoder_runs_sst2_adasvd_refactored_5budgets.csv"
    safe_remove "experiments/complete_e2e_retest.csv"
    safe_remove "experiments/comprehensive_test_results.csv"
    print_info "Backup files removed"

    echo ""
    print_info "Conservative cleanup complete!"
}

# ══════════════════════════════════════════════════════════════════════════
# Standard cleanup
# ══════════════════════════════════════════════════════════════════════════

cleanup_standard() {
    cleanup_conservative

    echo ""
    print_header "Standard cleanup (additional steps)"

    echo "Removing old training logs (keeping latest 3)..."
    if [ -d "experiments/logs" ]; then
        ls -t experiments/logs/*.log 2>/dev/null | tail -n +4 | xargs -r rm -f
        print_info "Old logs removed (kept latest 3)"
    fi

    echo ""
    echo "Removing old CSV results..."
    if [ -d "experiments" ]; then
        ls -t experiments/*.csv 2>/dev/null | grep -v "encoder_runs.csv" | tail -n +3 | xargs -r rm -f
        print_info "Old CSVs removed (kept latest and encoder_runs.csv)"
    fi

    echo ""
    print_info "Standard cleanup complete!"
}

# ══════════════════════════════════════════════════════════════════════════
# Deep cleanup
# ══════════════════════════════════════════════════════════════════════════

cleanup_deep() {
    cleanup_standard

    echo ""
    print_header "Deep cleanup (additional steps)"

    print_warning "About to remove all compressed models and logs!"
    echo ""
    echo "Will remove:"
    echo "  - compressed_models/ (compressed model checkpoints)"
    echo "  - pretrained/ (local pretrained models)"
    echo "  - experiments/logs/ (all log files)"
    echo ""
    echo "Set CONFIRM_DEEP=yes to skip this prompt."
    if [ "${CONFIRM_DEEP:-}" != "yes" ]; then
        print_warning "Deep cleanup skipped (set CONFIRM_DEEP=yes to proceed non-interactively)"
        return
    fi

    echo ""
    echo "Removing all models..."
    safe_remove "compressed_models"
    safe_remove "pretrained"

    echo ""
    echo "Removing all logs..."
    safe_remove "experiments/logs"
    mkdir -p experiments/logs  # Recreate empty directory

    echo ""
    print_info "Deep cleanup complete!"
}

# ══════════════════════════════════════════════════════════════════════════
# Main
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
print_header "Cleanup complete"

echo "Directory size after cleanup:"
du -sh . 2>/dev/null || echo "Unable to calculate size"
echo ""

echo "To view changes:"
echo "  git status"
echo ""
