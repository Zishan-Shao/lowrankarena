#!/bin/bash
# Test FlashSVD with realistic small ranks for actual compression
set -e

cd "$(dirname "$0")/../.."

MODEL="textattack/bert-base-uncased-SST-2"
TASK="sst2"
RANKS=(32 64 128 256)
OUT_CSV="eval_results/encoder_runs_small_ranks_comparison.csv"

echo "=========================================="
echo "Small Rank Comparison Test"
echo "=========================================="
echo ""
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Ranks: ${RANKS[@]}"
echo "Methods: SVD, FWSVD"
echo "Backends: naive, flashsvd"
echo "Output: $OUT_CSV"
echo ""

TOTAL_RUNS=$((${#RANKS[@]} * 2 * 2))  # 4 ranks × 2 methods × 2 backends = 16
CURRENT_RUN=0

for RANK in "${RANKS[@]}"; do
    for METHOD in svd fwsvd; do
        for BACKEND in naive flashsvd; do
            CURRENT_RUN=$((CURRENT_RUN + 1))
            
            echo "[$CURRENT_RUN/$TOTAL_RUNS] $METHOD rank=$RANK backend=$BACKEND"
            echo "-----------------------------------"
            
            if [ "$METHOD" = "fwsvd" ]; then
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

echo "=========================================="
echo "Small Rank Test Completed!"
echo "=========================================="
echo ""

# Show results
if [ -f "$OUT_CSV" ]; then
    echo "Results Summary:"
    echo "-----------------------------------"
    tail -n +2 "$OUT_CSV" | awk -F',' '{
        printf "%-6s rank=%-3s %-8s | Acc=%5.2f%% | Mem=%6s MB | Lat=%7s ms | Params=%5.2f%%\n",
        $9, $10, $13, $22*100, $25, $23, $26*100
    }' | column -t
    echo ""
fi

echo "✅ Done!"
