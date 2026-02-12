#!/bin/bash
# Complete AdaSVD + FlashSVD test with adaptive median rank strategy
# Budget < 0.3: auto median rank
# Budget >= 0.3: per-op adaptive ranks
set -e

cd "$(dirname "$0")/../.."

MODEL="textattack/bert-base-uncased-SST-2"
TASK="sst2"
SEQ_LEN=128
BATCH_SIZE=32

OUT_CSV="eval_results/final/adasvd_flashsvd_complete.csv"

echo "=========================================="
echo "AdaSVD + FlashSVD Complete Benchmark"
echo "=========================================="
echo ""
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Config: seq=$SEQ_LEN, batch=$BATCH_SIZE"
echo ""
echo "Testing budgets: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7"
echo "Backend: FlashSVD (Triton kernels)"
echo "Strategy:"
echo "  - Budget < 0.3: Auto median rank (compatibility)"
echo "  - Budget >= 0.3: Per-op adaptive ranks (performance)"
echo ""

# Remove old output CSV
rm -f "$OUT_CSV"

BUDGETS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7)
TOTAL_RUNS=${#BUDGETS[@]}
CURRENT_RUN=0

for BUDGET in "${BUDGETS[@]}"; do
    CURRENT_RUN=$((CURRENT_RUN + 1))

    echo "[$CURRENT_RUN/$TOTAL_RUNS] AdaSVD budget=$BUDGET backend=flashsvd"
    echo "-----------------------------------"

    python run_encoder_benchmark.py \
        --method adasvd \
        --budget $BUDGET \
        --backend flashsvd \
        --model_id "$MODEL" \
        --task $TASK \
        --seq_len $SEQ_LEN \
        --batch_size $BATCH_SIZE \
        --dtype fp16 \
        --calib_batches 4 \
        --seed 0 \
        --out_csv "$OUT_CSV" \
        --notes "AdaSVD FlashSVD budget=$BUDGET"

    echo ""
    echo "[$CURRENT_RUN/$TOTAL_RUNS] ✅ Completed"
    echo ""

    # Show budget report
    if [ -f "ars_out/budget_report.json" ]; then
        echo "Budget Report:"
        python3 -c "import json; r=json.load(open('ars_out/budget_report.json')); print(f\"  Target={r['target_budget']:.1f}, Achieved={r['achieved_ratio']:.3f}\")"
        echo ""
    fi

    # Archive ars_out
    if [ -d "ars_out" ]; then
        mkdir -p eval_results/archived_adasvd_outputs
        mv ars_out eval_results/archived_adasvd_outputs/ars_out_flashsvd_b${BUDGET}
    fi
done

echo "=========================================="
echo "AdaSVD + FlashSVD Complete Test DONE!"
echo "=========================================="
echo ""

if [ -f "$OUT_CSV" ]; then
    echo "Results Summary:"
    echo "-----------------------------------"
    echo "Budget | Accuracy | Latency | Throughput | Memory | Params% | Strategy"
    echo "-------|----------|---------|------------|--------|---------|----------"
    tail -n +2 "$OUT_CSV" | awk -F',' '{
        budget = $11
        acc = $22 * 100
        lat = $23
        thr = $24
        mem = $25
        prm = $26 * 100
        strategy = (budget < 0.3) ? "median" : "per-op"
        printf "%5.1f%% | %7.2f%% | %6.1f ms | %8.1f sps | %5.0f MB | %6.2f%% | %s\n",
            budget*100, acc, lat, thr, mem, prm, strategy
    }'
    echo ""

    echo "Performance Analysis:"
    echo "-----------------------------------"
    tail -n +2 "$OUT_CSV" | awk -F',' '{
        budget = $11
        lat = $23
        thr = $24
        strategy = (budget < 0.3) ? "median" : "per-op"
        printf "Budget=%5.1f%% %-8s | Latency=%6.1f ms | Throughput=%6.1f sps\n",
            budget*100, strategy, lat, thr
    }'
    echo ""
fi

echo "✅ All tests complete!"
echo "Results saved to: $OUT_CSV"
echo ""
echo "Next steps:"
echo "  1. Copy to eval_results/final/adasvd_benchmarks.csv"
echo "  2. Update all_encoder_benchmarks.csv"
echo "  3. Update MEMORY.md with median rank strategy findings"
