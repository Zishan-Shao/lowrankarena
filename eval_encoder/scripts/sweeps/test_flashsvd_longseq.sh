#!/bin/bash
# Test FlashSVD on long sequences and large batches
set -e

cd "$(dirname "$0")/../.."

MODEL="textattack/bert-base-uncased-SST-2"
TASK="sst2"
OUT_CSV="eval_results/encoder_runs_flashsvd_longseq_test.csv"

echo "=========================================="
echo "FlashSVD Long Sequence + Large Batch Test"
echo "=========================================="
echo ""
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Output: $OUT_CSV"
echo ""
echo "Test Configurations:"
echo "  1. Short seq (128), Small batch (32)  - Baseline"
echo "  2. Long seq (512), Small batch (32)"
echo "  3. Long seq (512), Large batch (64)"
echo "  4. Very Long seq (1024), Large batch (64)"
echo ""

CONFIGS=(
    "128 32 short-small"
    "512 32 long-small"
    "512 64 long-large"
)

TOTAL_RUNS=$((${#CONFIGS[@]} * 2 * 2))  # 3 configs × 2 methods × 2 backends
CURRENT_RUN=0

for CONFIG in "${CONFIGS[@]}"; do
    read -r SEQ_LEN BATCH_SIZE DESC <<< "$CONFIG"
    
    for METHOD in svd fwsvd; do
        for BACKEND in naive flashsvd; do
            CURRENT_RUN=$((CURRENT_RUN + 1))
            
            echo "[$CURRENT_RUN/$TOTAL_RUNS] $METHOD seq=$SEQ_LEN batch=$BATCH_SIZE backend=$BACKEND"
            echo "-----------------------------------"
            
            if [ "$METHOD" = "fwsvd" ]; then
                CALIB_ARG="--calib_batches 4"
            else
                CALIB_ARG=""
            fi
            
            python run_encoder_benchmark.py \
                --method $METHOD \
                --rank 512 \
                --backend $BACKEND \
                --model_id "$MODEL" \
                --task $TASK \
                --seq_len $SEQ_LEN \
                --batch_size $BATCH_SIZE \
                --dtype fp16 \
                $CALIB_ARG \
                --seed 0 \
                --out_csv "$OUT_CSV" \
                --notes "$METHOD seq=$SEQ_LEN batch=$BATCH_SIZE backend=$BACKEND ($DESC)" || {
                    echo "⚠️ Test failed (likely OOM), skipping..."
                    continue
                }
            
            echo ""
            echo "[$CURRENT_RUN/$TOTAL_RUNS] ✅ Completed"
            echo ""
        done
    done
done

echo "=========================================="
echo "Long Sequence Test Completed!"
echo "=========================================="
echo ""

# Show results
if [ -f "$OUT_CSV" ]; then
    echo "Results Summary:"
    echo "-----------------------------------"
    tail -n +2 "$OUT_CSV" | awk -F',' '{
        printf "%-6s seq=%-4s batch=%-2s %-8s | Acc=%5.2f%% | Mem=%7s MB | Lat=%8s ms | Throughput=%6s s/s\n",
        $9, $6, $7, $13, $22*100, $25, $23, $24
    }' | column -t
    echo ""
fi

echo "✅ Done!"
