#!/bin/bash
# Cross-layer Basis Sharing — Llama-3.1-8B-Instruct
# Keep ratios: 0.8 0.7 0.6 0.5 0.4
# Calib is built once (keep=0.8) and reused for all ratios.
# Model saved as HF safetensors directly (no .pt → convert step needed).
#
# Usage:
#   bash run_compress_llama31_8b_instruct_bs_cl.sh

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BS_DIR="$REPO_ROOT/lowrankarena/compress/svd/Basis_Sharing"
CFG_DIR="$BS_DIR/tasks/configs/wikitext_ppl/llama/share2"
OUTPUT_DIR="/home/ww247/lowrankarena/hf_ckpts/LowRankArena/llama31_8b_instruct/BasisSharingCL"
LOG_DIR="$BS_DIR/logs"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

cd "$BS_DIR"

for RATIO in 20 30 40 50 60; do
    KEEP=$(python3 -c "print(round(1 - $RATIO/100, 1))")
    CFG="$CFG_DIR/share_llama31_8b_instruct_${RATIO}_server.yaml"
    CKPT_DIR="$OUTPUT_DIR/llama31_8b_instruct_cl_bs_${KEEP}"
    LOG="$LOG_DIR/llama31_8b_instruct_cl_bs_${KEEP}.log"

    if [ -d "$CKPT_DIR" ] && [ -f "$CKPT_DIR/config.json" ]; then
        echo "=== checkpoint exists, skipping: $CKPT_DIR ==="
        continue
    fi

    echo "=== Cross-layer BS keep=$KEEP (compression_ratio=$RATIO) ==="
    conda run -n lowrank_compress --no-capture-output \
        python test.py --cf "$CFG" \
        2>&1 | tee "$LOG"
done

echo ""
echo "=== All done ==="
for RATIO in 20 30 40 50 60; do
    KEEP=$(python3 -c "print(round(1 - $RATIO/100, 1))")
    CKPT_DIR="$OUTPUT_DIR/llama31_8b_instruct_cl_bs_${KEEP}"
    [ -f "$CKPT_DIR/config.json" ] && echo "  ✓ $CKPT_DIR" || echo "  ✗ MISSING: $CKPT_DIR"
done
