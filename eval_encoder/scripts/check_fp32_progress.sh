#!/bin/bash
# Check progress of fp32 benchmark suite

OUTPUT_FILE="/tmp/claude-1000/-mnt-e-learning-SVD-Benchmark-lowrankarena/tasks/b1a61a6.output"
CSV_FILE="/mnt/e/learning/SVD-Benchmark/lowrankarena/eval_encoder/eval_results/encoder_runs.csv"

echo "==================================="
echo "fp32 Benchmark Progress"
echo "==================================="
echo ""

# Check if process is running
if pgrep -f "run_complete_fp32_benchmark.sh" > /dev/null; then
    echo "Status: ✓ RUNNING"
else
    echo "Status: ✗ NOT RUNNING or COMPLETED"
fi

echo ""

# Count completed tests from output
COMPLETED=$(grep -c "✓ Test.*completed" "$OUTPUT_FILE" 2>/dev/null || echo "0")
echo "Tests completed: $COMPLETED / 36"

# Show current test
echo ""
echo "Current test:"
grep "\[.*\] Running:" "$OUTPUT_FILE" 2>/dev/null | tail -1 || echo "No test running"

# Show last few lines
echo ""
echo "Recent output (last 15 lines):"
echo "-----------------------------------"
tail -15 "$OUTPUT_FILE" 2>/dev/null || echo "No output yet"
