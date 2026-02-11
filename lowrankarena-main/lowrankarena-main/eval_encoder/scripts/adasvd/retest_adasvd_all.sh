#!/bin/bash
# Re-test ALL AdaSVD configurations with FIXED budget control
set -e

cd "$(dirname "$0")/../.."

MODEL="textattack/bert-base-uncased-SST-2"
TASK="sst2"
SEQ_LEN=128
BATCH_SIZE=32

OUT_CSV="eval_results/final/adasvd_benchmarks_FIXED.csv"
BACKUP_CSV="eval_results/final/adasvd_benchmarks_BROKEN_BACKUP.csv"

echo "=========================================="
echo "AdaSVD Complete Re-test (FIXED)"
echo "=========================================="
echo ""
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Config: seq=$SEQ_LEN, batch=$BATCH_SIZE"
echo ""
echo "Testing budgets: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7"
echo "Backends: naive ONLY (FlashSVD incompatible with per-op adaptive ranks)"
echo "Total: 7 tests"
echo ""
echo "Bug fixes applied:"
echo "  1. Budget base: SVD params → Original model params"
echo "  2. Budget loss: One-sided log → Two-sided squared error"
echo "  3. Loss coefficients: lambda=100.0, gamma=0.01"
echo ""

# Backup old broken data
if [ -f "eval_results/final/adasvd_benchmarks.csv" ]; then
    echo "Backing up old BROKEN data to: $BACKUP_CSV"
    cp eval_results/final/adasvd_benchmarks.csv "$BACKUP_CSV"
    echo ""
fi

# Remove old output CSV if exists
rm -f "$OUT_CSV"

BUDGETS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7)
BACKENDS=(naive)  # FlashSVD incompatible with per-operation adaptive ranks

TOTAL_RUNS=$((${#BUDGETS[@]} * ${#BACKENDS[@]}))
CURRENT_RUN=0

for BUDGET in "${BUDGETS[@]}"; do
    for BACKEND in "${BACKENDS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))

        echo "[$CURRENT_RUN/$TOTAL_RUNS] AdaSVD budget=$BUDGET backend=$BACKEND"
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
            --notes "AdaSVD FIXED budget=$BUDGET backend=$BACKEND"

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
            mkdir -p eval_results/archived_adasvd_outputs
            mv ars_out eval_results/archived_adasvd_outputs/ars_out_FIXED_b${BUDGET}_${BACKEND}
        fi
    done
done

echo "=========================================="
echo "AdaSVD Complete Re-test DONE!"
echo "=========================================="
echo ""

if [ -f "$OUT_CSV" ]; then
    echo "Results Summary:"
    echo "-----------------------------------"
    tail -n +2 "$OUT_CSV" | awk -F',' '{
        printf "Budget=%-3s %-8s | Acc=%5.2f%% | Mem=%6s MB | Lat=%7s ms | Params=%5.2f%%\n",
        $11, $13, $22*100, $25, $23, $26*100
    }'
    echo ""

    echo "Budget Control Verification:"
    echo "-----------------------------------"
    tail -n +2 "$OUT_CSV" | awk -F',' '{
        target = $11
        actual = $26
        diff = (actual - target) / target * 100
        status = (diff < 5 && diff > -5) ? "✅ OK" : "❌ FAIL"
        printf "Budget=%-3s %-8s | Target=%5.1f%% | Actual=%5.2f%% | Diff=%+6.1f%% | %s\n",
            target, $13, target*100, actual*100, diff, status
    }' | sort -u
    echo ""
fi

echo "✅ All tests complete!"
echo "Results saved to: $OUT_CSV"
echo "Broken data backed up to: $BACKUP_CSV"
echo ""
echo "Next steps:"
echo "  1. Replace old adasvd_benchmarks.csv with FIXED version"
echo "  2. Update all_encoder_benchmarks.csv"
echo "  3. Update small_ranks_complete_benchmark.csv"
echo "  4. Update all reports and documentation"
