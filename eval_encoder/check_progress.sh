#!/bin/bash
# Monitor test progress

echo "=== Test Progress Monitor ==="
echo ""

# Count completed tests
TOTAL_TESTS=9
CSV_FILE="eval_results/comprehensive_test_results.csv"

if [ -f "$CSV_FILE" ]; then
    COMPLETED=$(($(wc -l < "$CSV_FILE") - 1))  # Subtract header
    echo "Completed: $COMPLETED / $TOTAL_TESTS tests"
    echo ""

    if [ $COMPLETED -gt 0 ]; then
        echo "Latest results:"
        tail -3 "$CSV_FILE" | cut -d, -f1,9,11,22,25,26,27 | column -t -s,
        echo ""
        echo "Columns: timestamp, method, budget, metric_value, peak_infer_mb, peak_e2e_mb, peak_mb"
    fi
else
    echo "No results yet..."
fi

echo ""
echo "To view full results:"
echo "  cat $CSV_FILE | column -t -s,"
echo ""
echo "To monitor in real-time:"
echo "  watch -n 5 bash check_progress.sh"
