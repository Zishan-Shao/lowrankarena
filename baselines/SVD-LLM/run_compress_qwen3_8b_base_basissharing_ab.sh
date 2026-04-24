#!/bin/bash
# ABLinear Basis Sharing compression for Qwen/Qwen3-8B-Base
# Uses compress_qwen3_lowrank.py (LowRankQwen3ForCausalLM, ABLinear schema)
# Keep ratios: 0.8 0.7 0.6 0.5 0.4
# Outputs HF-compatible safetensors directly (no .pt → convert step).
#
# Usage:
#   bash run_compress_qwen3_8b_base_basissharing_ab.sh

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LRA_DIR="$REPO_ROOT/lowrankarena/lowrankarena"
MODEL="Qwen/Qwen3-8B-Base"
MODEL_TAG="Qwen3-8B-Base"
OUTPUT_DIR="/home/ww247/lowrankarena/hf_ckpts/LowRankArena/qwen3_8b_base/BasisSharingAB"
LOG_DIR="$LRA_DIR/logs"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

for KEEP in 0.8 0.7 0.6 0.5 0.4; do
    CKPT_DIR="$OUTPUT_DIR/qwen3_8b_base_bs_ab_${KEEP}"
    LOG="$LOG_DIR/${MODEL_TAG}_bs_ab_${KEEP}.log"

    if [ -d "$CKPT_DIR" ] && [ -f "$CKPT_DIR/config.json" ]; then
        echo "=== checkpoint exists, skipping: $CKPT_DIR ==="
        continue
    fi

    echo "=== ABLinear BasisSharing keep=$KEEP ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    conda run -n lowrank_compress --no-capture-output \
        python "$LRA_DIR/scripts/compress_qwen3_lowrank.py" \
            --model-id "$MODEL" \
            --method basis_sharing \
            --keep-ratio "$KEEP" \
            --output-dir "$CKPT_DIR" \
            --device cuda:0 \
            --dataset wikitext2 \
            --calibration-samples 64 \
            --sequence-length 2048 \
        2>&1 | tee "$LOG"
done

echo ""
echo "=== All done ==="
for KEEP in 0.8 0.7 0.6 0.5 0.4; do
    CKPT_DIR="$OUTPUT_DIR/qwen3_8b_base_bs_ab_${KEEP}"
    [ -f "$CKPT_DIR/config.json" ] && echo "  ✓ $CKPT_DIR" || echo "  ✗ MISSING: $CKPT_DIR"
done
