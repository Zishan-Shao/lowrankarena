#!/bin/bash
# Retest Priority 1: AdaSVD budgets + Long sequences with E2E memory tracking
# Estimated time: ~40 minutes

set -e

MODEL="textattack/bert-base-uncased-SST-2"
TASK="sst2"
BATCH_SIZE=32
DTYPE="fp16"
OUT_CSV="eval_results/e2e_priority1_retest.csv"

echo "=========================================="
echo "Priority 1 Retest: E2E Memory Data"
echo "Output: $OUT_CSV"
echo "=========================================="

# ============================================
# 1. AdaSVD Multi-Budget Tests (10 tests)
# ============================================
echo ""
echo "[1/2] Testing AdaSVD with 5 budgets..."

for BUDGET in 0.1 0.2 0.3 0.5 0.7; do
    echo ""
    echo "--- AdaSVD budget=$BUDGET (naive) ---"
    python run_encoder_benchmark.py \
        --model_id "$MODEL" \
        --task "$TASK" \
        --method adasvd \
        --budget $BUDGET \
        --seq_len 128 \
        --batch_size $BATCH_SIZE \
        --dtype $DTYPE \
        --backend naive \
        --out_csv "$OUT_CSV" \
        --notes "AdaSVD budget=$BUDGET naive (e2e retest)"

    echo ""
    echo "--- AdaSVD budget=$BUDGET (flashsvd) ---"
    python run_encoder_benchmark.py \
        --model_id "$MODEL" \
        --task "$TASK" \
        --method adasvd \
        --budget $BUDGET \
        --seq_len 128 \
        --batch_size $BATCH_SIZE \
        --dtype $DTYPE \
        --backend flashsvd \
        --out_csv "$OUT_CSV" \
        --notes "AdaSVD budget=$BUDGET flashsvd (e2e retest)"
done

# ============================================
# 2. Long Sequence Tests (12 tests)
# ============================================
echo ""
echo "[2/2] Testing long sequences (256, 512, 1024)..."

for SEQ_LEN in 256 512 1024; do
    for METHOD in svd fwsvd drone adasvd; do
        if [ "$METHOD" = "adasvd" ]; then
            RANK_OR_BUDGET="--budget 0.3"
            NOTE_SUFFIX="budget=0.3"
        else
            RANK_OR_BUDGET="--rank 128"
            NOTE_SUFFIX="rank=128"
        fi

        echo ""
        echo "--- $METHOD seq_len=$SEQ_LEN ---"
        python run_encoder_benchmark.py \
            --model_id "$MODEL" \
            --task "$TASK" \
            --method $METHOD \
            $RANK_OR_BUDGET \
            --seq_len $SEQ_LEN \
            --batch_size $BATCH_SIZE \
            --dtype $DTYPE \
            --backend naive \
            --out_csv "$OUT_CSV" \
            --notes "$METHOD $NOTE_SUFFIX seq_len=$SEQ_LEN (e2e retest)"
    done
done

echo ""
echo "=========================================="
echo "Priority 1 Retest Complete!"
echo "Results saved to: $OUT_CSV"
echo "=========================================="
