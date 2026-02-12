#!/bin/bash
# Check progress of AdaSVD benchmark

# Change to eval_encoder directory
cd "$(dirname "$0")/../.."

LOG_FILE="scripts/adasvd/test_adasvd_5budgets.log"
CSV_FILE="eval_results/encoder_runs_sst2_adasvd_refactored_5budgets.csv"

echo "=========================================="
echo "AdaSVD Benchmark Progress"
echo "=========================================="
echo ""

if [ -f "$LOG_FILE" ]; then
    # Count completed runs
    COMPLETED=$(grep -c "✅ Completed" "$LOG_FILE" || echo "0")
    echo "Completed runs: $COMPLETED / 10"
    echo ""

    # Show last few lines of log
    echo "Last 20 lines of log:"
    echo "-----------------------------------"
    tail -n 20 "$LOG_FILE"
    echo ""
fi

if [ -f "$CSV_FILE" ]; then
    # Show results so far
    echo "Results so far:"
    echo "-----------------------------------"
    tail -n +2 "$CSV_FILE" | grep "adasvd" | \
        awk -F',' '{printf "Budget=%.1f %-8s | Acc=%s | Mem=%s MB | Lat=%s ms\n",
                    $11, $13, $22, $25, $23}' | column -t
    echo ""
fi

# Check if still running
if pgrep -f "run_encoder_benchmark.py" > /dev/null; then
    echo "Status: 🔄 RUNNING"
    echo ""
    echo "To follow live output:"
    echo "  tail -f $LOG_FILE"
else
    echo "Status: ✅ COMPLETED (or not started)"
fi

echo ""
