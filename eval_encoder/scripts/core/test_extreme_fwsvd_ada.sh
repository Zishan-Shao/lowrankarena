#!/bin/bash
# Extreme Memory Test: FWSVD and AdaSVD at seq=512, batch=64
set -e

cd "$(dirname "$0")/../.."

MODEL="textattack/bert-base-uncased-SST-2"
TASK="sst2"
SEQ_LEN=512
BATCH_SIZE=64

OUT_CSV="eval_results/final/extreme_memory_benchmark.csv"

echo "=========================================="
echo "Extreme Memory: FWSVD + AdaSVD"
echo "=========================================="
echo ""
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Config: seq=$SEQ_LEN, batch=$BATCH_SIZE"
echo ""
echo "Testing FWSVD rank=32 and AdaSVD budget=0.1"
echo "Output: $OUT_CSV"
echo ""

TOTAL_RUNS=6  # FWSVD (2 backends) + AdaSVD (2 backends, 2 budgets)
CURRENT_RUN=0

# FWSVD rank=32
echo "=========================================="
echo "FWSVD rank=32"
echo "=========================================="
echo ""

for BACKEND in naive flashsvd; do
    CURRENT_RUN=$((CURRENT_RUN + 1))

    echo "[$CURRENT_RUN/$TOTAL_RUNS] fwsvd rank=32 seq=$SEQ_LEN batch=$BATCH_SIZE backend=$BACKEND"
    echo "-----------------------------------"

    python run_encoder_benchmark.py \
        --method fwsvd \
        --rank 32 \
        --backend $BACKEND \
        --model_id "$MODEL" \
        --task $TASK \
        --seq_len $SEQ_LEN \
        --batch_size $BATCH_SIZE \
        --dtype fp16 \
        --calib_batches 4 \
        --seed 0 \
        --out_csv "$OUT_CSV" \
        --notes "fwsvd rank=32 seq=$SEQ_LEN batch=$BATCH_SIZE backend=$BACKEND"

    echo ""
    echo "[$CURRENT_RUN/$TOTAL_RUNS] ✅ Completed"
    echo ""
done

# AdaSVD budget=0.1 and 0.3
echo "=========================================="
echo "AdaSVD (budgets 0.1, 0.3)"
echo "=========================================="
echo ""

for BUDGET in 0.1 0.3; do
    for BACKEND in naive flashsvd; do
        CURRENT_RUN=$((CURRENT_RUN + 1))

        echo "[$CURRENT_RUN/$TOTAL_RUNS] adasvd budget=$BUDGET seq=$SEQ_LEN batch=$BATCH_SIZE backend=$BACKEND"
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
            --notes "adasvd budget=$BUDGET seq=$SEQ_LEN batch=$BATCH_SIZE backend=$BACKEND"

        echo ""
        echo "[$CURRENT_RUN/$TOTAL_RUNS] ✅ Completed"
        echo ""

        # Archive ars_out
        if [ -d "ars_out" ]; then
            mkdir -p eval_results/archived_adasvd_outputs
            mv ars_out eval_results/archived_adasvd_outputs/ars_out_extreme_b${BUDGET}_${BACKEND}
        fi
    done
done

echo "=========================================="
echo "FWSVD + AdaSVD Test Finished!"
echo "=========================================="
echo ""

if [ -f "$OUT_CSV" ]; then
    echo "All Results:"
    echo "-----------------------------------"
    tail -n +2 "$OUT_CSV" | awk -F',' '{
        printf "%-6s %-4s %-8s | Acc=%5.2f%% | Mem=%7s MB | Lat=%7s ms | Params=%5.2f%%\n",
        $9, ($10!="" ? "r="$10 : "b="$11), $13, $22*100, $25, $23, $26*100
    }' | tail -10
    echo ""
    echo "Memory Savings Summary:"
    echo "-----------------------------------"
    tail -n +2 "$OUT_CSV" | awk -F',' '
    {
        key = $9 "," ($10!="" ? $10 : $11)
        backend = $13
        mem[key,backend] = $25
        lat[key,backend] = $23
        params[key] = $26
    }
    END {
        for (combo in mem) {
            split(combo, parts, SUBSEP)
            if (parts[2] == "naive") {
                key = parts[1]
                naive_mem = mem[combo]
                naive_lat = lat[combo]
                flash_mem = mem[key,"flashsvd"]
                flash_lat = lat[key,"flashsvd"]
                if (flash_mem != "") {
                    mem_save = (naive_mem - flash_mem) / naive_mem * 100
                    lat_ratio = flash_lat / naive_lat
                    printf "%-15s | Naive=%6s MB  Flash=%6s MB  Save=%5.1f%%  Speed=%.2fx  Params=%.1f%%\n",
                        key, naive_mem, flash_mem, mem_save, lat_ratio, params[key]*100
                }
            }
        }
    }' | sort
    echo ""
fi

echo "✅ All tests complete!"
