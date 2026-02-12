#!/bin/bash
# Benchmark AdaSVD with 5 budgets (0.3, 0.4, 0.5, 0.6, 0.7)
# Tests both naive and FlashSVD backends

set -e  # Exit on error

# Change to eval_encoder directory
cd "$(dirname "$0")/../.."

MODEL="textattack/bert-base-uncased-SST-2"
TASK="sst2"
BUDGETS=(0.3 0.4 0.5 0.6 0.7)
BACKENDS=("naive" "flashsvd")

OUT_CSV="eval_results/encoder_runs_sst2_adasvd_refactored_5budgets.csv"

echo "=========================================="
echo "AdaSVD Benchmark: 5 Budgets × 2 Backends"
echo "=========================================="
echo ""
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Budgets: ${BUDGETS[@]}"
echo "Backends: ${BACKENDS[@]}"
echo "Output: $OUT_CSV"
echo ""
echo "Total runs: $(( ${#BUDGETS[@]} * ${#BACKENDS[@]} )) = ${#BUDGETS[@]} budgets × ${#BACKENDS[@]} backends"
echo ""

TOTAL_RUNS=$(( ${#BUDGETS[@]} * ${#BACKENDS[@]} ))
CURRENT_RUN=0

for BACKEND in "${BACKENDS[@]}"; do
    echo "=========================================="
    echo "Backend: $BACKEND"
    echo "=========================================="
    echo ""

    for BUDGET in "${BUDGETS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))

        echo "[$CURRENT_RUN/$TOTAL_RUNS] AdaSVD budget=$BUDGET backend=$BACKEND"
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
            --notes "AdaSVD refactored budget=$BUDGET backend=$BACKEND"

        echo ""
        echo "[$CURRENT_RUN/$TOTAL_RUNS] ✅ Completed"
        echo ""

        # Clean up ars_out to avoid conflicts between runs
        if [ -d "ars_out" ]; then
            echo "Archiving ars_out to ars_out_budget${BUDGET}_${BACKEND}..."
            mv ars_out ars_out_budget${BUDGET}_${BACKEND}
        fi
    done
done

echo "=========================================="
echo "All Tests Completed!"
echo "=========================================="
echo ""
echo "Results saved to: $OUT_CSV"
echo ""

# Show results summary
echo "Results Summary:"
echo "-----------------------------------"
tail -n +2 "$OUT_CSV" | grep "adasvd" | \
    awk -F',' '{printf "Budget=%.1f %-8s | Acc=%s | Mem=%s MB | Lat=%s ms | Params=%s\n",
                $11, $13, $22, $25, $23, $26}' | column -t

echo ""
echo "Expected improvements vs old AdaSVD:"
echo "  Old (MaskedSVDLinear): 673 MB, 53.29 ms"
echo "  New (naive):           ~300 MB, ~45 ms"
echo "  New (flashsvd):        ~300 MB, ~40 ms"
echo ""

# Generate comparison report
echo "Generating detailed comparison report..."
python -c "
import csv
import sys

# Read results
results = []
with open('$OUT_CSV', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['method'] == 'adasvd':
            results.append(row)

# Group by backend
naive_results = [r for r in results if r['backend'] == 'naive']
flash_results = [r for r in results if r['backend'] == 'flashsvd']

print()
print('=' * 80)
print('Detailed Comparison: Naive vs FlashSVD')
print('=' * 80)
print()
print('Budget | Naive Mem | Flash Mem | Mem Δ | Naive Lat | Flash Lat | Lat Δ | Accuracy')
print('-' * 80)

for budget in [0.3, 0.4, 0.5, 0.6, 0.7]:
    naive = [r for r in naive_results if float(r.get('budget', 0)) == budget]
    flash = [r for r in flash_results if float(r.get('budget', 0)) == budget]

    if naive and flash:
        n = naive[0]
        f = flash[0]
        n_mem = float(n['peak_mem_mb'])
        f_mem = float(f['peak_mem_mb'])
        mem_delta = ((f_mem - n_mem) / n_mem) * 100

        n_lat = float(n['latency_ms'])
        f_lat = float(f['latency_ms'])
        lat_delta = ((f_lat - n_lat) / n_lat) * 100

        acc = float(n['metric_value'])

        print(f'{budget:>5.1f} | {n_mem:>9.1f} | {f_mem:>9.1f} | {mem_delta:>5.1f}% | {n_lat:>9.2f} | {f_lat:>9.2f} | {lat_delta:>5.1f}% | {acc:>8.4f}')

print()
print('Legend:')
print('  Mem Δ: (Flash - Naive) / Naive × 100%  (negative = FlashSVD uses less memory)')
print('  Lat Δ: (Flash - Naive) / Naive × 100%  (negative = FlashSVD is faster)')
print()
"

echo "✅ Benchmark completed successfully!"
