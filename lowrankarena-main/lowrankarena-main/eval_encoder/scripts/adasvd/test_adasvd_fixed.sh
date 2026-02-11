#!/bin/bash
# Test AdaSVD with fixed budget control
set -e

cd "$(dirname "$0")/../.."

MODEL="textattack/bert-base-uncased-SST-2"
TASK="sst2"
SEQ_LEN=128
BATCH_SIZE=32

OUT_CSV="eval_results/final/adasvd_fixed_test.csv"

echo "=========================================="
echo "AdaSVD Fixed Budget Control Test"
echo "=========================================="
echo ""
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Config: seq=$SEQ_LEN, batch=$BATCH_SIZE"
echo ""
echo "Testing budgets: 0.3, 0.5"
echo "Expected: Ratio should converge to target!"
echo ""
echo "Fixes applied:"
echo "  1. Budget base: SVD params → Original model params"
echo "  2. Budget loss: One-sided log → Two-sided squared error"
echo "  3. Loss coefficients: lambda=100.0, gamma=0.01"
echo ""

TOTAL_RUNS=4  # 2 budgets × 2 backends
CURRENT_RUN=0

for BUDGET in 0.3 0.5; do
    for BACKEND in naive flashsvd; do
        CURRENT_RUN=$((CURRENT_RUN + 1))

        echo "[$CURRENT_RUN/$TOTAL_RUNS] adasvd budget=$BUDGET backend=$BACKEND"
        echo "-----------------------------------"

        python run_encoder_benchmark.py \
            --method adasvd \
            --budget $BUDGET \
            --backend $BACKEND \
            --model_id "$MODEL" \
            --task $TASK \
            --seq_len $SEQ_LEN \
            --batch_size $BATCH_SIZE \
            --dtype fp16 \
            --calib_batches 4 \
            --seed 0 \
            --out_csv "$OUT_CSV" \
            --notes "adasvd budget=$BUDGET backend=$BACKEND (FIXED)"

        echo ""
        echo "[$CURRENT_RUN/$TOTAL_RUNS] ✅ Completed"
        echo ""

        # Show budget report
        if [ -f "ars_out/budget_report.json" ]; then
            echo "Budget Report:"
            cat ars_out/budget_report.json | python3 -m json.tool | grep -E "(target_budget|achieved_ratio)"
            echo ""
        fi

        # Archive ars_out
        if [ -d "ars_out" ]; then
            mv ars_out eval_results/archived_adasvd_outputs/ars_out_fixed_b${BUDGET}_${BACKEND}
        fi
    done
done

echo "=========================================="
echo "AdaSVD Fixed Test Complete!"
echo "=========================================="
echo ""

if [ -f "$OUT_CSV" ]; then
    echo "Results:"
    echo "-----------------------------------"
    tail -n +2 "$OUT_CSV" | awk -F',' '{
        printf "Budget=%-3s %-8s | Acc=%5.2f%% | Mem=%6s MB | Lat=%7s ms | Params=%5.2f%%\n",
        $11, $13, $22*100, $25, $23, $26*100
    }'
    echo ""

    echo "Budget Control Check:"
    echo "-----------------------------------"
    tail -n +2 "$OUT_CSV" | awk -F',' '{
        target = $11
        actual = $26
        diff = (actual - target) / target * 100
        status = (diff < 20 && diff > -20) ? "✅ OK" : "❌ FAIL"
        printf "Budget=%-3s | Target=%5.1f%% | Actual=%5.2f%% | Diff=%+6.1f%% | %s\n",
            target, target*100, actual*100, diff, status
    }' | sort -u
    echo ""
fi

echo "✅ Test complete!"
echo "Results saved to: $OUT_CSV"
echo ""
echo "Expected outcome:"
echo "  ✅ Budget 0.3 → Actual ~30% params (not 66.5%!)"
echo "  ✅ Budget 0.5 → Actual ~50% params (not 66.5%!)"
