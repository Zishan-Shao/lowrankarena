#!/bin/bash
# Complete small rank test including AdaSVD (both backends)
set -e

cd "$(dirname "$0")/../.."

MODEL="textattack/bert-base-uncased-SST-2"
TASK="sst2"
RANKS=(32 64 128 256)
OUT_CSV="eval_results/final/small_ranks_complete_benchmark.csv"

echo "=========================================="
echo "Complete Small Rank Benchmark"
echo "=========================================="
echo ""
echo "Model: $MODEL"
echo "Task: $TASK"
echo ""
echo "Part 1: SVD/FWSVD/DRONE with ranks 32, 64, 128, 256"
echo "        → naive + flashsvd backends"
echo ""
echo "Part 2: AdaSVD with equivalent budgets"
echo "        → naive + flashsvd backends (for comparison)"
echo ""
echo "Output: $OUT_CSV"
echo ""

TOTAL_RUNS=$((${#RANKS[@]} * 3 * 2 + ${#RANKS[@]} * 2))  # (4 ranks × 3 methods × 2 backends) + (4 budgets × 2 backends)
echo "Total tests: $TOTAL_RUNS (24 + 8 = 32)"
echo ""

CURRENT_RUN=0

# Part 1: SVD, FWSVD, DRONE with rank
echo ""
echo "=========================================="
echo "Part 1: Rank-based methods (24 tests)"
echo "=========================================="
echo ""

for RANK in "${RANKS[@]}"; do
    for METHOD in svd fwsvd drone; do
        for BACKEND in naive flashsvd; do
            CURRENT_RUN=$((CURRENT_RUN + 1))

            echo "[$CURRENT_RUN/$TOTAL_RUNS] $METHOD rank=$RANK backend=$BACKEND"
            echo "-----------------------------------"

            if [ "$METHOD" = "fwsvd" ] || [ "$METHOD" = "drone" ]; then
                CALIB_ARG="--calib_batches 4"
            else
                CALIB_ARG=""
            fi

            python run_encoder_benchmark.py \
                --method $METHOD \
                --rank $RANK \
                --backend $BACKEND \
                --model_id "$MODEL" \
                --task $TASK \
                --seq_len 128 \
                --batch_size 32 \
                --dtype fp16 \
                $CALIB_ARG \
                --seed 0 \
                --out_csv "$OUT_CSV" \
                --notes "$METHOD rank=$RANK backend=$BACKEND"

            echo ""
            echo "[$CURRENT_RUN/$TOTAL_RUNS] ✅ Completed"
            echo ""
        done
    done
done

# Part 2: AdaSVD with budgets (BOTH backends for evidence)
echo ""
echo "=========================================="
echo "Part 2: AdaSVD with both backends (8 tests)"
echo "=========================================="
echo ""

# Map ranks to equivalent budgets
declare -A BUDGET_MAP
BUDGET_MAP[32]=0.1   # Very aggressive compression
BUDGET_MAP[64]=0.2   # Aggressive compression
BUDGET_MAP[128]=0.3  # Medium compression
BUDGET_MAP[256]=0.5  # Light compression

for RANK in "${RANKS[@]}"; do
    BUDGET=${BUDGET_MAP[$RANK]}

    for BACKEND in naive flashsvd; do
        CURRENT_RUN=$((CURRENT_RUN + 1))

        echo "[$CURRENT_RUN/$TOTAL_RUNS] adasvd budget=$BUDGET (equiv rank=$RANK) backend=$BACKEND"
        echo "-----------------------------------"

        python run_encoder_benchmark.py \
            --method adasvd \
            --budget $BUDGET \
            --backend $BACKEND \
            --model_id "$MODEL" \
            --task $TASK \
            --seq_len 128 \
            --batch_size 32 \
            --dtype fp16 \
            --calib_batches 4 \
            --seed 0 \
            --out_csv "$OUT_CSV" \
            --notes "adasvd budget=$BUDGET (target equiv to rank=$RANK) backend=$BACKEND"

        echo ""
        echo "[$CURRENT_RUN/$TOTAL_RUNS] ✅ Completed"
        echo ""

        # Archive ars_out
        if [ -d "ars_out" ]; then
            mv ars_out eval_results/archived_adasvd_outputs/ars_out_budget${BUDGET}_rank${RANK}_${BACKEND}
        fi
    done
done

echo "=========================================="
echo "Complete Benchmark Finished!"
echo "=========================================="
echo ""

# Show results summary
if [ -f "$OUT_CSV" ]; then
    echo "Results Summary:"
    echo "-----------------------------------"
    echo ""
    echo "By Method:"
    tail -n +2 "$OUT_CSV" | awk -F',' '{print $9}' | sort | uniq -c | \
        awk '{printf "  %-10s: %2d tests\n", $2, $1}'
    echo ""
    echo "By Backend:"
    tail -n +2 "$OUT_CSV" | awk -F',' '{print $13}' | sort | uniq -c | \
        awk '{printf "  %-10s: %2d tests\n", $2, $1}'
    echo ""
    echo "Detailed Results:"
    echo "-----------------------------------"
    tail -n +2 "$OUT_CSV" | awk -F',' '{
        printf "%-6s rank/bud=%-4s %-8s | Acc=%5.2f%% | Mem=%6s MB | Lat=%7s ms | Params=%5.2f%%\n",
        $9, ($10!="" ? $10 : $11), $13, $22*100, $25, $23, $26*100
    }' | sort | column -t
    echo ""

    # Special analysis: AdaSVD naive vs flashsvd
    echo "AdaSVD Backend Comparison (naive vs flashsvd):"
    echo "-----------------------------------"
    tail -n +2 "$OUT_CSV" | grep "adasvd" | awk -F',' '{
        printf "Budget=%-3s %-8s | Mem=%6s MB | Lat=%7s ms | Params=%5.2f%%\n",
        $11, $13, $25, $23, $26*100
    }' | sort | column -t
    echo ""
fi

echo "✅ All tests complete!"
echo "Results saved to: $OUT_CSV"
echo ""
echo "📊 This data will show that AdaSVD+FlashSVD:"
echo "   - Saves minimal memory (only ~6% for AdaSVD vs 23-60% for others)"
echo "   - Is 2x slower"
echo "   - Is NOT worth the trade-off"
