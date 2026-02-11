#!/bin/bash
# Long Sequence + Low Rank Test: Show FlashSVD's memory advantage in extreme scenarios
set -e

cd "$(dirname "$0")/../.."

MODEL="textattack/bert-base-uncased-SST-2"
TASK="sst2"

echo "=========================================="
echo "Long Sequence + Low Rank Benchmark"
echo "=========================================="
echo ""
echo "Model: $MODEL"
echo "Task: $TASK"
echo ""
echo "Test 1: Long Sequences (seq=512, batch=32)"
echo "Test 2: Large Batches (seq=128, batch=128)"
echo "Test 3: Extreme (seq=1024, batch=16)"
echo ""
echo "Ranks: 32, 64 (aggressive compression)"
echo "Methods: SVD, DRONE (best performers)"
echo "Backends: naive, flashsvd"
echo ""

OUT_CSV="eval_results/final/longseq_lowrank_benchmark.csv"
echo "Output: $OUT_CSV"
echo ""

TOTAL_RUNS=24  # 3 configs × 2 ranks × 2 methods × 2 backends = 24
CURRENT_RUN=0

# Test 1: Long Sequences (seq=512, batch=32)
echo ""
echo "=========================================="
echo "Test 1: Long Sequences (seq=512, batch=32)"
echo "=========================================="
echo ""

for RANK in 32 64; do
    for METHOD in svd drone; do
        for BACKEND in naive flashsvd; do
            CURRENT_RUN=$((CURRENT_RUN + 1))

            echo "[$CURRENT_RUN/$TOTAL_RUNS] $METHOD rank=$RANK seq=512 batch=32 backend=$BACKEND"
            echo "-----------------------------------"

            if [ "$METHOD" = "drone" ]; then
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
                --seq_len 512 \
                --batch_size 32 \
                --dtype fp16 \
                $CALIB_ARG \
                --seed 0 \
                --out_csv "$OUT_CSV" \
                --notes "$METHOD rank=$RANK seq=512 batch=32 backend=$BACKEND"

            echo ""
            echo "[$CURRENT_RUN/$TOTAL_RUNS] ✅ Completed"
            echo ""
        done
    done
done

# Test 2: Large Batches (seq=128, batch=128)
echo ""
echo "=========================================="
echo "Test 2: Large Batches (seq=128, batch=128)"
echo "=========================================="
echo ""

for RANK in 32 64; do
    for METHOD in svd drone; do
        for BACKEND in naive flashsvd; do
            CURRENT_RUN=$((CURRENT_RUN + 1))

            echo "[$CURRENT_RUN/$TOTAL_RUNS] $METHOD rank=$RANK seq=128 batch=128 backend=$BACKEND"
            echo "-----------------------------------"

            if [ "$METHOD" = "drone" ]; then
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
                --batch_size 128 \
                --dtype fp16 \
                $CALIB_ARG \
                --seed 0 \
                --out_csv "$OUT_CSV" \
                --notes "$METHOD rank=$RANK seq=128 batch=128 backend=$BACKEND"

            echo ""
            echo "[$CURRENT_RUN/$TOTAL_RUNS] ✅ Completed"
            echo ""
        done
    done
done

# Test 3: Extreme (seq=1024, batch=16)
echo ""
echo "=========================================="
echo "Test 3: Extreme (seq=1024, batch=16)"
echo "=========================================="
echo ""

for RANK in 32 64; do
    for METHOD in svd drone; do
        for BACKEND in naive flashsvd; do
            CURRENT_RUN=$((CURRENT_RUN + 1))

            echo "[$CURRENT_RUN/$TOTAL_RUNS] $METHOD rank=$RANK seq=1024 batch=16 backend=$BACKEND"
            echo "-----------------------------------"

            if [ "$METHOD" = "drone" ]; then
                CALIB_ARG="--calib_batches 2"  # Fewer batches for very long sequences
            else
                CALIB_ARG=""
            fi

            python run_encoder_benchmark.py \
                --method $METHOD \
                --rank $RANK \
                --backend $BACKEND \
                --model_id "$MODEL" \
                --task $TASK \
                --seq_len 1024 \
                --batch_size 16 \
                --dtype fp16 \
                $CALIB_ARG \
                --seed 0 \
                --out_csv "$OUT_CSV" \
                --notes "$METHOD rank=$RANK seq=1024 batch=16 backend=$BACKEND"

            echo ""
            echo "[$CURRENT_RUN/$TOTAL_RUNS] ✅ Completed"
            echo ""
        done
    done
done

echo "=========================================="
echo "Long Sequence Benchmark Finished!"
echo "=========================================="
echo ""

# Show results summary
if [ -f "$OUT_CSV" ]; then
    echo "Results Summary:"
    echo "-----------------------------------"
    echo ""
    echo "By Configuration:"
    tail -n +2 "$OUT_CSV" | awk -F',' '{print $6":"$7}' | sort | uniq -c | \
        awk '{printf "  seq=%s batch=%s: %2d tests\n", substr($2,1,index($2,":")-1), substr($2,index($2,":")+1), $1}'
    echo ""
    echo "Memory Savings (FlashSVD vs Naive):"
    echo "-----------------------------------"
    tail -n +2 "$OUT_CSV" | awk -F',' '
    {
        key = $9 "," $10 "," $6 "," $7
        mem[key,$13] = $25
    }
    END {
        for (combo in mem) {
            split(combo, parts, SUBSEP)
            if (parts[2] == "naive") {
                naive = mem[combo]
                flash_key = parts[1] SUBSEP "flashsvd"
                if (flash_key in mem) {
                    flash = mem[flash_key]
                    saving = (naive - flash) / naive * 100
                    split(parts[1], info, ",")
                    printf "%-6s rank=%-3s seq=%-4s batch=%-3s | Naive=%5s MB  Flash=%5s MB  Save=%5.1f%%\n",
                        info[1], info[2], info[3], info[4], naive, flash, saving
                }
            }
        }
    }' | sort
    echo ""
fi

echo "✅ All tests complete!"
echo "Results saved to: $OUT_CSV"
echo ""
echo "📊 Expected findings:"
echo "   - Long sequences: 50-70% memory savings with FlashSVD"
echo "   - Large batches: 40-60% memory savings"
echo "   - Extreme (1024): 60-80% memory savings"
echo "   - FlashSVD's value increases with memory pressure!"
echo ""
