#!/bin/bash
# Complete E2E Memory Retest - All 37 old tests
# Estimated time: ~90-120 minutes

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flashsvd

MODEL="textattack/bert-base-uncased-SST-2"
TASK="sst2"
BATCH_SIZE=32
DTYPE="fp16"
OUT_CSV="eval_results/complete_e2e_retest.csv"

echo "=========================================="
echo "Complete E2E Memory Retest"
echo "Output: $OUT_CSV"
echo "Estimated time: ~90-120 minutes"
echo "=========================================="

# Function to run test with error handling
run_test() {
    local desc="$1"
    shift
    echo ""
    echo ">>> Testing: $desc"
    if ! python run_encoder_benchmark.py "$@" --out_csv "$OUT_CSV"; then
        echo "❌ Failed: $desc"
        return 1
    fi
    echo "✅ Completed: $desc"
}

# ============================================
# Part 1: AdaSVD Multi-Budget Tests
# From: encoder_runs_sst2_adasvd_refactored_5budgets.csv
# Tests: 10 (5 budgets × 2 backends)
# ============================================
echo ""
echo "=========================================="
echo "Part 1/5: AdaSVD Multi-Budget (10 tests)"
echo "=========================================="

for BUDGET in 0.3 0.4 0.5 0.6 0.7; do
    run_test "AdaSVD budget=$BUDGET naive" \
        --model_id "$MODEL" --task "$TASK" \
        --method adasvd --budget $BUDGET \
        --seq_len 128 --batch_size $BATCH_SIZE --dtype $DTYPE \
        --backend naive \
        --notes "AdaSVD budget=$BUDGET naive (complete retest)"

    run_test "AdaSVD budget=$BUDGET flashsvd" \
        --model_id "$MODEL" --task "$TASK" \
        --method adasvd --budget $BUDGET \
        --seq_len 128 --batch_size $BATCH_SIZE --dtype $DTYPE \
        --backend flashsvd \
        --notes "AdaSVD budget=$BUDGET flashsvd (complete retest)"
done

# ============================================
# Part 2: Additional AdaSVD Tests
# From: encoder_runs.csv
# Tests: 8 (various budgets including 0.1, 0.2)
# ============================================
echo ""
echo "=========================================="
echo "Part 2/5: Additional AdaSVD (8 tests)"
echo "=========================================="

# Tests from encoder_runs.csv (unique ones not in Part 1)
for BUDGET in 0.1 0.2; do
    run_test "AdaSVD budget=$BUDGET naive" \
        --model_id "$MODEL" --task "$TASK" \
        --method adasvd --budget $BUDGET \
        --seq_len 128 --batch_size $BATCH_SIZE --dtype $DTYPE \
        --backend naive \
        --notes "AdaSVD budget=$BUDGET naive (encoder_runs retest)"

    run_test "AdaSVD budget=$BUDGET flashsvd" \
        --model_id "$MODEL" --task "$TASK" \
        --method adasvd --budget $BUDGET \
        --seq_len 128 --batch_size $BATCH_SIZE --dtype $DTYPE \
        --backend flashsvd \
        --notes "AdaSVD budget=$BUDGET flashsvd (encoder_runs retest)"
done

# Additional from encoder_runs.csv (0.4 flashsvd already in Part 1)
run_test "AdaSVD budget=0.3 naive (dup)" \
    --model_id "$MODEL" --task "$TASK" \
    --method adasvd --budget 0.3 \
    --seq_len 128 --batch_size $BATCH_SIZE --dtype $DTYPE \
    --backend naive \
    --notes "AdaSVD budget=0.3 naive (encoder_runs duplicate)"

run_test "AdaSVD budget=0.5 naive (dup)" \
    --model_id "$MODEL" --task "$TASK" \
    --method adasvd --budget 0.5 \
    --seq_len 128 --batch_size $BATCH_SIZE --dtype $DTYPE \
    --backend naive \
    --notes "AdaSVD budget=0.5 naive (encoder_runs duplicate)"

run_test "AdaSVD budget=0.3 flashsvd (dup)" \
    --model_id "$MODEL" --task "$TASK" \
    --method adasvd --budget 0.3 \
    --seq_len 128 --batch_size $BATCH_SIZE --dtype $DTYPE \
    --backend flashsvd \
    --notes "AdaSVD budget=0.3 flashsvd (encoder_runs duplicate)"

run_test "AdaSVD budget=0.5 flashsvd (dup)" \
    --model_id "$MODEL" --task "$TASK" \
    --method adasvd --budget 0.5 \
    --seq_len 128 --batch_size $BATCH_SIZE --dtype $DTYPE \
    --backend flashsvd \
    --notes "AdaSVD budget=0.5 flashsvd (encoder_runs duplicate)"

# ============================================
# Part 3: Long Sequence Tests
# From: encoder_runs_flashsvd_longseq_test.csv
# Tests: 8 (2 seq_lens × 2 methods × 2 backends)
# Note: Original has duplicates at seq_len=512, we only run once
# ============================================
echo ""
echo "=========================================="
echo "Part 3/5: Long Sequence Tests (8 tests)"
echo "=========================================="

for SEQ_LEN in 256 512; do
    for METHOD in svd fwsvd; do
        RANK=512

        run_test "$METHOD rank=$RANK seq_len=$SEQ_LEN naive" \
            --model_id "$MODEL" --task "$TASK" \
            --method $METHOD --rank $RANK \
            --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --dtype $DTYPE \
            --backend naive \
            --notes "$METHOD rank=$RANK seq_len=$SEQ_LEN naive (longseq retest)"

        run_test "$METHOD rank=$RANK seq_len=$SEQ_LEN flashsvd" \
            --model_id "$MODEL" --task "$TASK" \
            --method $METHOD --rank $RANK \
            --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --dtype $DTYPE \
            --backend flashsvd \
            --notes "$METHOD rank=$RANK seq_len=$SEQ_LEN flashsvd (longseq retest)"
    done
done

# ============================================
# Part 4: FlashSVD Comparison (seq_len=128)
# From: encoder_runs_flashsvd_comparison.csv
# Tests: 4 (2 methods × 2 backends)
# ============================================
echo ""
echo "=========================================="
echo "Part 4/5: FlashSVD Comparison (4 tests)"
echo "=========================================="

for METHOD in svd fwsvd; do
    RANK=512

    run_test "$METHOD rank=$RANK naive" \
        --model_id "$MODEL" --task "$TASK" \
        --method $METHOD --rank $RANK \
        --seq_len 128 --batch_size $BATCH_SIZE --dtype $DTYPE \
        --backend naive \
        --notes "$METHOD rank=$RANK naive (flashsvd comparison)"

    run_test "$METHOD rank=$RANK flashsvd" \
        --model_id "$MODEL" --task "$TASK" \
        --method $METHOD --rank $RANK \
        --seq_len 128 --batch_size $BATCH_SIZE --dtype $DTYPE \
        --backend flashsvd \
        --notes "$METHOD rank=$RANK flashsvd (flashsvd comparison)"
done

# ============================================
# Part 5: Small Ranks Comparison
# From: encoder_runs_small_ranks_comparison.csv
# Tests: 3 (2 methods × 1.5 backends - one missing)
# ============================================
echo ""
echo "=========================================="
echo "Part 5/5: Small Ranks Comparison (3 tests)"
echo "=========================================="

RANK=32

run_test "svd rank=$RANK naive" \
    --model_id "$MODEL" --task "$TASK" \
    --method svd --rank $RANK \
    --seq_len 128 --batch_size $BATCH_SIZE --dtype $DTYPE \
    --backend naive \
    --notes "svd rank=$RANK naive (small rank)"

run_test "svd rank=$RANK flashsvd" \
    --model_id "$MODEL" --task "$TASK" \
    --method svd --rank $RANK \
    --seq_len 128 --batch_size $BATCH_SIZE --dtype $DTYPE \
    --backend flashsvd \
    --notes "svd rank=$RANK flashsvd (small rank)"

run_test "fwsvd rank=$RANK naive" \
    --model_id "$MODEL" --task "$TASK" \
    --method fwsvd --rank $RANK \
    --seq_len 128 --batch_size $BATCH_SIZE --dtype $DTYPE \
    --backend naive \
    --notes "fwsvd rank=$RANK naive (small rank)"

# Note: Original CSV doesn't have fwsvd rank=32 flashsvd,
# but we can add it for completeness
run_test "fwsvd rank=$RANK flashsvd (bonus)" \
    --model_id "$MODEL" --task "$TASK" \
    --method fwsvd --rank $RANK \
    --seq_len 128 --batch_size $BATCH_SIZE --dtype $DTYPE \
    --backend flashsvd \
    --notes "fwsvd rank=$RANK flashsvd (small rank, bonus test)"

# ============================================
# Summary
# ============================================
echo ""
echo "=========================================="
echo "Complete E2E Retest Finished!"
echo "=========================================="
echo ""
echo "Results saved to: $OUT_CSV"
echo ""
echo "Test Summary:"
echo "  Part 1: AdaSVD Multi-Budget     - 10 tests ✅"
echo "  Part 2: Additional AdaSVD       -  8 tests ✅"
echo "  Part 3: Long Sequence Tests     -  8 tests ✅"
echo "  Part 4: FlashSVD Comparison     -  4 tests ✅"
echo "  Part 5: Small Ranks Comparison  -  4 tests ✅"
echo "  ────────────────────────────────────────────"
echo "  Total:                           34 tests ✅"
echo ""
echo "Note: Some tests from encoder_runs.csv are duplicates"
echo "      of tests in other CSVs, so actual unique tests"
echo "      may be fewer than original 37."
echo ""
echo "Next steps:"
echo "  1. Review results: head -20 $OUT_CSV"
echo "  2. Merge with comprehensive_test_results.csv"
echo "  3. Generate analysis report"
echo ""
