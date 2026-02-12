#!/bin/bash
# Quick progress check for E2E retest

LOG_FILE="retest_all_e2e.log"
CSV_FILE="eval_results/complete_e2e_retest.csv"

echo "=========================================="
echo "E2E Retest Progress"
echo "=========================================="
echo ""

# Check if test is running
if pgrep -f "retest_all_e2e.sh" > /dev/null; then
    echo "✅ Test is RUNNING"
else
    echo "⏸️  Test is NOT running (completed or stopped)"
fi
echo ""

# Count completed tests
COMPLETED=$(grep -c "✅ Completed" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Tests completed: $COMPLETED / 34"
echo "Progress: $(echo "scale=1; $COMPLETED * 100 / 34" | bc)%"
echo ""

# Show current test
echo "Current test:"
grep ">>> Testing" "$LOG_FILE" 2>/dev/null | tail -1
echo ""

# Show recent completions
echo "Recent completions:"
grep "✅ Completed" "$LOG_FILE" 2>/dev/null | tail -3
echo ""

# Check for failures
FAILED=$(grep -c "❌ Failed" "$LOG_FILE" 2>/dev/null || echo "0")
if [ "$FAILED" -gt 0 ]; then
    echo "⚠️  Failed tests: $FAILED"
    grep "❌ Failed" "$LOG_FILE" 2>/dev/null
    echo ""
fi

# CSV progress
if [ -f "$CSV_FILE" ]; then
    CSV_LINES=$(wc -l < "$CSV_FILE")
    echo "CSV rows written: $((CSV_LINES - 1)) (excluding header)"
else
    echo "CSV not yet created"
fi
echo ""

# Estimate time remaining
if [ "$COMPLETED" -gt 0 ]; then
    REMAINING=$((34 - COMPLETED))
    AVG_TIME=3  # minutes per test
    EST_MIN=$((REMAINING * AVG_TIME))
    echo "Estimated time remaining: ~$EST_MIN minutes"
fi

echo ""
echo "=========================================="
