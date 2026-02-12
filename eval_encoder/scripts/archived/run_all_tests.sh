#!/bin/bash
#
# Comprehensive test suite for eval_encoder with Peak Memory enhancement
# Tests all compression methods with different backends and configurations
#

set -e  # Exit on error

# Configuration
MODEL="textattack/bert-base-uncased-SST-2"
TASK="sst2"
SEQ_LEN=128
BATCH_SIZE=32
MEASURE_STEPS=20
WARMUP_STEPS=5
CALIB_BATCHES=4
OUT_CSV="eval_results/comprehensive_test_results.csv"

# Clean up old results
rm -f "$OUT_CSV"
echo "Starting comprehensive test suite..."
echo "Results will be saved to: $OUT_CSV"
echo ""

# Activate environment
PYTHON="/home/wwh/miniconda3/envs/flashsvd/bin/python"

# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Dense baseline (no compression)
# ═══════════════════════════════════════════════════════════════════════════
echo "=========================================="
echo "Test 1/9: Dense (baseline)"
echo "=========================================="
$PYTHON run_encoder_benchmark.py \
  --model_id=$MODEL \
  --task=$TASK \
  --method=dense \
  --backend=naive \
  --seq_len=$SEQ_LEN \
  --batch_size=$BATCH_SIZE \
  --measure_steps=$MEASURE_STEPS \
  --warmup_steps=$WARMUP_STEPS \
  --out_csv=$OUT_CSV \
  --notes="Dense baseline"

# ═══════════════════════════════════════════════════════════════════════════
# Test 2: SVD with naive backend (rank=64)
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "=========================================="
echo "Test 2/9: SVD (rank=64, naive)"
echo "=========================================="
$PYTHON run_encoder_benchmark.py \
  --model_id=$MODEL \
  --task=$TASK \
  --method=svd \
  --rank=64 \
  --backend=naive \
  --seq_len=$SEQ_LEN \
  --batch_size=$BATCH_SIZE \
  --measure_steps=$MEASURE_STEPS \
  --warmup_steps=$WARMUP_STEPS \
  --out_csv=$OUT_CSV \
  --notes="SVD rank=64 naive"

# ═══════════════════════════════════════════════════════════════════════════
# Test 3: SVD with naive backend (rank=128)
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "=========================================="
echo "Test 3/9: SVD (rank=128, naive)"
echo "=========================================="
$PYTHON run_encoder_benchmark.py \
  --model_id=$MODEL \
  --task=$TASK \
  --method=svd \
  --rank=128 \
  --backend=naive \
  --seq_len=$SEQ_LEN \
  --batch_size=$BATCH_SIZE \
  --measure_steps=$MEASURE_STEPS \
  --warmup_steps=$WARMUP_STEPS \
  --out_csv=$OUT_CSV \
  --notes="SVD rank=128 naive"

# ═══════════════════════════════════════════════════════════════════════════
# Test 4: SVD with FlashSVD backend (rank=128)
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "=========================================="
echo "Test 4/9: SVD (rank=128, flashsvd)"
echo "=========================================="
$PYTHON run_encoder_benchmark.py \
  --model_id=$MODEL \
  --task=$TASK \
  --method=svd \
  --rank=128 \
  --backend=flashsvd \
  --seq_len=$SEQ_LEN \
  --batch_size=$BATCH_SIZE \
  --measure_steps=$MEASURE_STEPS \
  --warmup_steps=$WARMUP_STEPS \
  --out_csv=$OUT_CSV \
  --notes="SVD rank=128 flashsvd"

# ═══════════════════════════════════════════════════════════════════════════
# Test 5: FWSVD with naive backend (rank=128)
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "=========================================="
echo "Test 5/9: FWSVD (rank=128, naive)"
echo "=========================================="
$PYTHON run_encoder_benchmark.py \
  --model_id=$MODEL \
  --task=$TASK \
  --method=fwsvd \
  --rank=128 \
  --backend=naive \
  --seq_len=$SEQ_LEN \
  --batch_size=$BATCH_SIZE \
  --measure_steps=$MEASURE_STEPS \
  --warmup_steps=$WARMUP_STEPS \
  --calib_batches=$CALIB_BATCHES \
  --out_csv=$OUT_CSV \
  --notes="FWSVD rank=128 naive"

# ═══════════════════════════════════════════════════════════════════════════
# Test 6: DRONE with naive backend (rank=128)
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "=========================================="
echo "Test 6/9: DRONE (rank=128, naive)"
echo "=========================================="
$PYTHON run_encoder_benchmark.py \
  --model_id=$MODEL \
  --task=$TASK \
  --method=drone \
  --rank=128 \
  --backend=naive \
  --seq_len=$SEQ_LEN \
  --batch_size=$BATCH_SIZE \
  --measure_steps=$MEASURE_STEPS \
  --warmup_steps=$WARMUP_STEPS \
  --calib_batches=$CALIB_BATCHES \
  --out_csv=$OUT_CSV \
  --notes="DRONE rank=128 naive"

# ═══════════════════════════════════════════════════════════════════════════
# Test 7: AdaSVD with naive backend (budget=0.2)
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "=========================================="
echo "Test 7/9: AdaSVD (budget=0.2, naive)"
echo "=========================================="
$PYTHON run_encoder_benchmark.py \
  --model_id=$MODEL \
  --task=$TASK \
  --method=adasvd \
  --budget=0.2 \
  --backend=naive \
  --seq_len=$SEQ_LEN \
  --batch_size=$BATCH_SIZE \
  --measure_steps=$MEASURE_STEPS \
  --warmup_steps=$WARMUP_STEPS \
  --calib_batches=$CALIB_BATCHES \
  --out_csv=$OUT_CSV \
  --notes="AdaSVD budget=0.2 naive"

# ═══════════════════════════════════════════════════════════════════════════
# Test 8: AdaSVD with naive backend (budget=0.3)
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "=========================================="
echo "Test 8/9: AdaSVD (budget=0.3, naive)"
echo "=========================================="
$PYTHON run_encoder_benchmark.py \
  --model_id=$MODEL \
  --task=$TASK \
  --method=adasvd \
  --budget=0.3 \
  --backend=naive \
  --seq_len=$SEQ_LEN \
  --batch_size=$BATCH_SIZE \
  --measure_steps=$MEASURE_STEPS \
  --warmup_steps=$WARMUP_STEPS \
  --calib_batches=$CALIB_BATCHES \
  --out_csv=$OUT_CSV \
  --notes="AdaSVD budget=0.3 naive"

# ═══════════════════════════════════════════════════════════════════════════
# Test 9: AdaSVD with FlashSVD backend (budget=0.3)
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "=========================================="
echo "Test 9/9: AdaSVD (budget=0.3, flashsvd)"
echo "=========================================="
$PYTHON run_encoder_benchmark.py \
  --model_id=$MODEL \
  --task=$TASK \
  --method=adasvd \
  --budget=0.3 \
  --backend=flashsvd \
  --seq_len=$SEQ_LEN \
  --batch_size=$BATCH_SIZE \
  --measure_steps=$MEASURE_STEPS \
  --warmup_steps=$WARMUP_STEPS \
  --calib_batches=$CALIB_BATCHES \
  --out_csv=$OUT_CSV \
  --notes="AdaSVD budget=0.3 flashsvd"

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
echo ""
echo "Results saved to: $OUT_CSV"
echo ""
echo "Summary:"
head -1 "$OUT_CSV"
tail -n +2 "$OUT_CSV" | column -t -s,

echo ""
echo "To view detailed results:"
echo "  cat $OUT_CSV | column -t -s,"
echo ""
echo "To extract peak memory columns:"
echo "  cat $OUT_CSV | cut -d, -f1,9,24,25,26"
echo "  # Fields: timestamp, method, peak_mem_infer_mb, peak_mem_e2e_mb, peak_mem_mb"
