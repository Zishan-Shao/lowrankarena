#!/bin/bash
# SVD-LLM Basis Sharing (step 5) — GQA-aware
# Model: meta-llama/Llama-3.1-8B-Instruct
# Keep ratios: 0.8 0.7 0.6 0.5 0.4
# SVDLLM.py saves files as _basis_sharing_{KEEP}.pt (not reduction ratio).
#
# Usage:
#   bash run_compress_llama31_8b_instruct_bs.sh [HF_TOKEN]

set -eo pipefail
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_TAG="Llama-3.1-8B-Instruct"
MODEL_PREFIX="meta_llama_Llama_3.1_8B_Instruct"
SAVE_DIR="checkpoints/svdllm/llama31_8b_instruct"
OUTPUT_DIR="/home/ww247/lowrankarena/hf_ckpts/LowRankArena/llama31_8b_instruct/BasisSharing"
HF_TOKEN="${1:-}"
SEQ_LEN=2048
PROF_MAT="$SAVE_DIR/${MODEL_PREFIX}_profiling_wikitext2_256_0.pt"

mkdir -p "$SAVE_DIR" "$OUTPUT_DIR" logs

TOKEN_ARG=""
[ -n "$HF_TOKEN" ] && TOKEN_ARG="--hf_token $HF_TOKEN"

# ── Step 1: Compress ──────────────────────────────────────────────────────────
for KEEP in 0.8 0.7 0.6 0.5 0.4; do
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_basis_sharing_${KEEP}.pt"
    if [ -f "$CKPT" ]; then
        echo "=== checkpoint exists, skipping: $CKPT ==="
        continue
    fi

    RATIO=$(python -c "print(round(1 - $KEEP, 1))")
    echo "=== Compress BasisSharing keep=$KEEP (--ratio=$RATIO) ==="
    PROF_ARG=""
    [ -f "$PROF_MAT" ] && PROF_ARG="--profiling_mat_path $PROF_MAT"
    python SVDLLM.py --model "$MODEL" --step 5 --ratio $RATIO \
        $PROF_ARG \
        --save_path "$SAVE_DIR" --model_seq_len $SEQ_LEN $TOKEN_ARG \
        2>&1 | tee "logs/${MODEL_TAG}_bs_${KEEP}.log"
done

# Collect existing checkpoints
CKPTS=""
for KEEP in 0.8 0.7 0.6 0.5 0.4; do
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_basis_sharing_${KEEP}.pt"
    [ -f "$CKPT" ] && CKPTS="$CKPTS $CKPT"
done

if [ -z "$CKPTS" ]; then
    echo "No checkpoints found, exiting."
    exit 1
fi

# ── Step 2: Strip RoPE cache ──────────────────────────────────────────────────
echo ""
echo "=== Stripping RoPE cache ==="
python strip_rope_cache.py $CKPTS \
    2>&1 | tee "logs/${MODEL_TAG}_bs_strip.log"

# ── Step 3: Convert to safetensors ────────────────────────────────────────────
echo ""
echo "=== Converting to safetensors → $OUTPUT_DIR ==="
for KEEP in 0.8 0.7 0.6 0.5 0.4; do
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_basis_sharing_${KEEP}.pt"
    ST_DIR="$OUTPUT_DIR/${MODEL_PREFIX}_basis_sharing_${KEEP}"
    [ ! -f "$CKPT" ] && continue
    if [ ! -f "$ST_DIR/model.safetensors" ]; then
        echo "  converting: $CKPT"
        rm -rf "$ST_DIR"
        TMPDIR_ST=$(mktemp -d)
        python convert_pt_to_safetensors.py "$CKPT" --output_dir "$TMPDIR_ST" \
            2>&1 | tee -a "logs/${MODEL_TAG}_bs_convert_st.log"
        SRC="$TMPDIR_ST/${MODEL_PREFIX}_basis_sharing_${KEEP}"
        if [ -d "$SRC" ]; then
            mv "$SRC" "$ST_DIR"
        else
            mv "$TMPDIR_ST"/*/ "$ST_DIR" 2>/dev/null || true
        fi
        rmdir "$TMPDIR_ST" 2>/dev/null || true
    else
        echo "  already converted: $ST_DIR/model.safetensors"
    fi
done

echo ""
echo "=== All done ==="
for KEEP in 0.8 0.7 0.6 0.5 0.4; do
    ST_DIR="$OUTPUT_DIR/${MODEL_PREFIX}_basis_sharing_${KEEP}"
    [ -f "$ST_DIR/model.safetensors" ] && echo "  ✓ $ST_DIR" || echo "  ✗ MISSING: $ST_DIR"
done
