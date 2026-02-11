#!/bin/bash
# Extreme Memory Pressure Test: seq=512, batch=64, rank=32
set -e

cd "$(dirname "$0")/../.."

MODEL="textattack/bert-base-uncased-SST-2"
TASK="sst2"
RANK=32
SEQ_LEN=512
BATCH_SIZE=64

OUT_CSV="eval_results/final/extreme_memory_benchmark.csv"

echo "=========================================="
echo "Extreme Memory Pressure Test"
echo "=========================================="
echo ""
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Config: seq=$SEQ_LEN, batch=$BATCH_SIZE, rank=$RANK"
echo ""
echo "Expected: Highest memory savings + best speed"
echo "Output: $OUT_CSV"
echo ""

TOTAL_RUNS=4  # 2 methods × 2 backends
CURRENT_RUN=0

for METHOD in svd drone; do
    for BACKEND in naive flashsvd; do
        CURRENT_RUN=$((CURRENT_RUN + 1))

        echo "[$CURRENT_RUN/$TOTAL_RUNS] $METHOD rank=$RANK seq=$SEQ_LEN batch=$BATCH_SIZE backend=$BACKEND"
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
            --seq_len $SEQ_LEN \
            --batch_size $BATCH_SIZE \
            --dtype fp16 \
            $CALIB_ARG \
            --seed 0 \
            --out_csv "$OUT_CSV" \
            --notes "$METHOD rank=$RANK seq=$SEQ_LEN batch=$BATCH_SIZE backend=$BACKEND"

        echo ""
        echo "[$CURRENT_RUN/$TOTAL_RUNS] ✅ Completed"
        echo ""
    done
done

echo "=========================================="
echo "Extreme Memory Test Finished!"
echo "=========================================="
echo ""

if [ -f "$OUT_CSV" ]; then
    echo "Results:"
    echo "-----------------------------------"
    tail -n +2 "$OUT_CSV" | awk -F',' '{
        printf "%-6s %-8s | Acc=%5.2f%% | Mem=%7s MB | Lat=%7s ms | Tput=%7s sps\n",
        $9, $13, $22*100, $25, $23, $24
    }'
    echo ""
    echo "Memory Savings:"
    echo "-----------------------------------"
    tail -n +2 "$OUT_CSV" | awk -F',' '
    {
        method = $9
        backend = $13
        mem[method,backend] = $25
        lat[method,backend] = $23
    }
    END {
        for (m in mem) {
            split(m, parts, SUBSEP)
            method = parts[1]
            backend = parts[2]
            if (backend == "naive") {
                naive_mem = mem[m]
                naive_lat = lat[m]
                flash_mem = mem[method,"flashsvd"]
                flash_lat = lat[method,"flashsvd"]
                mem_save = (naive_mem - flash_mem) / naive_mem * 100
                lat_ratio = flash_lat / naive_lat
                printf "%-6s | Naive=%6s MB  Flash=%6s MB  Save=%5.1f%%  |  Speed=%.2fx\n",
                    method, naive_mem, flash_mem, mem_save, lat_ratio
            }
        }
    }'
    echo ""
fi

echo "✅ Test complete!"
echo "📊 Expected: 70%+ memory savings, 0.7x-0.9x speed (faster!)"
