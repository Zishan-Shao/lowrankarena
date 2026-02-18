#!/bin/bash
################################################################################
# Setup Test Script - 验证环境是否就绪
################################################################################

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║              GLUE Benchmark Environment Test                       ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

FAILED=0

# Test 1: Python
echo -n "Testing Python... "
if python --version &> /dev/null; then
    echo "✓ $(python --version)"
else
    echo "✗ FAILED"
    FAILED=$((FAILED + 1))
fi

# Test 2: NVIDIA GPU
echo -n "Testing NVIDIA GPU... "
if nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "✓ $GPU_NAME"
else
    echo "✗ FAILED"
    FAILED=$((FAILED + 1))
fi

# Test 3: PyTorch
echo -n "Testing PyTorch... "
if python -c "import torch; print(f'v{torch.__version__}')" &> /dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "✓ v$TORCH_VERSION"
else
    echo "✗ FAILED"
    FAILED=$((FAILED + 1))
fi

# Test 4: CUDA
echo -n "Testing CUDA... "
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$CUDA_AVAILABLE" = "True" ]; then
    CUDA_DEVICE=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "✓ $CUDA_DEVICE"
else
    echo "✗ FAILED"
    FAILED=$((FAILED + 1))
fi

# Test 5: Transformers
echo -n "Testing Transformers... "
if python -c "import transformers; print(f'v{transformers.__version__}')" &> /dev/null; then
    TRANS_VERSION=$(python -c "import transformers; print(transformers.__version__)")
    echo "✓ v$TRANS_VERSION"
else
    echo "✗ FAILED"
    FAILED=$((FAILED + 1))
fi

# Test 6: Datasets
echo -n "Testing Datasets... "
if python -c "import datasets; print(f'v{datasets.__version__}')" &> /dev/null; then
    DATASETS_VERSION=$(python -c "import datasets; print(datasets.__version__)")
    echo "✓ v$DATASETS_VERSION"
else
    echo "✗ FAILED"
    FAILED=$((FAILED + 1))
fi

# Test 7: Evaluate
echo -n "Testing Evaluate... "
if python -c "import evaluate" &> /dev/null; then
    echo "✓"
else
    echo "✗ FAILED"
    FAILED=$((FAILED + 1))
fi

# Test 8: Required files
echo -n "Testing required files... "
if [ -f "eval_encoder/glue_pipeline.py" ] && \
   [ -f "eval_encoder/load_compressed_model.py" ] && \
   [ -f "eval_encoder/run_encoder_benchmark.py" ]; then
    echo "✓"
else
    echo "✗ FAILED (missing files)"
    FAILED=$((FAILED + 1))
fi

# Test 9: Write permissions
echo -n "Testing write permissions... "
TEST_DIR="eval_encoder/test_write_$$"
if mkdir -p "$TEST_DIR" 2>/dev/null && rmdir "$TEST_DIR" 2>/dev/null; then
    echo "✓"
else
    echo "✗ FAILED"
    FAILED=$((FAILED + 1))
fi

# Test 10: Quick GLUE data loading
echo -n "Testing GLUE data loading... "
if python -c "from datasets import load_dataset; ds = load_dataset('glue', 'sst2', split='validation[:10]')" &> /dev/null; then
    echo "✓"
else
    echo "✗ FAILED (network issue?)"
    FAILED=$((FAILED + 1))
fi

echo ""
echo "────────────────────────────────────────────────────────────────────"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ All tests passed! Environment is ready."
    echo ""
    echo "You can now run:"
    echo "  bash eval_encoder/scripts/one_click_glue.sh"
    exit 0
else
    echo "❌ $FAILED test(s) failed. Please fix the issues above."
    echo ""
    echo "Common fixes:"
    echo "  - Install PyTorch: pip install torch torchvision torchaudio"
    echo "  - Install packages: pip install transformers datasets evaluate scikit-learn scipy"
    echo "  - Check CUDA: nvidia-smi"
    exit 1
fi
