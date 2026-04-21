#!/bin/bash
# SVD-LLM V2 heterogeneous compression (fully consistent with new repo)
# Model: meta-llama/Llama-3.1-8B
# Ratios: 0.2 0.3 0.4 0.5 0.6
#
# Pipeline: compress → strip RoPE → convert to safetensors
#
# Usage:
#   bash run_compress_llama31_8b_v2.sh [HF_TOKEN]

set -eo pipefail
cd "$(dirname "$0")"

MODEL="meta-llama/Llama-3.1-8B"
MODEL_TAG="Llama-3.1-8B"
MODEL_PREFIX="meta_llama_Llama_3.1_8B"
SAVE_DIR="checkpoints/svdllm/llama31_8b"
OUTPUT_DIR="/home/ww247/lowrankarena/hf_ckpts/LowRankArena/LLama31_8b/SVDLLMv2"
HF_TOKEN="${1:-}"
SEQ_LEN=2048
SUFFIX="v2hetero"

PROF_MAT="$SAVE_DIR/${MODEL_PREFIX}_profiling_hetero_wikitext2_256_0.pt"

mkdir -p "$SAVE_DIR" "$OUTPUT_DIR" logs

TOKEN_ARG=""
[ -n "$HF_TOKEN" ] && TOKEN_ARG="--hf_token $HF_TOKEN"

if [ -f "$PROF_MAT" ]; then
    echo "=== Hetero profiling_mat found, reusing: $PROF_MAT ==="
    PROF_ARG="--profiling_mat_path $PROF_MAT"
else
    echo "=== Hetero profiling_mat not found, will reprofile ==="
    PROF_ARG="--reprofile --save_profiling_mat $PROF_MAT"
fi

# ── Step 1: Compress ──────────────────────────────────────────────────────────
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(1 - $RATIO)")
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_${SUFFIX}_${KEEP}.pt"

    if [ -f "$CKPT" ]; then
        echo "=== checkpoint exists, skipping: $CKPT ==="
        PROF_ARG="--profiling_mat_path $PROF_MAT"
        continue
    fi

    KEEP_DISPLAY=$(python -c "print(round(1 - $RATIO, 1))")
    echo "=== Compress ${SUFFIX} ratio=$RATIO (keep=$KEEP_DISPLAY) ==="
    python run_svdllm_v2_compress.py \
        --model "$MODEL" \
        --ratio $RATIO \
        $PROF_ARG \
        --save_path "$SAVE_DIR" \
        --model_seq_len $SEQ_LEN \
        $TOKEN_ARG \
        2>&1 | tee "logs/${MODEL_TAG}_${SUFFIX}_${KEEP_DISPLAY}.log"

    PROF_ARG="--profiling_mat_path $PROF_MAT"
done

# Collect existing checkpoints
CKPTS=""
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(1 - $RATIO)")
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_${SUFFIX}_${KEEP}.pt"
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
    2>&1 | tee "logs/${MODEL_TAG}_${SUFFIX}_strip.log"

# ── Step 3: Convert to safetensors ────────────────────────────────────────────
echo ""
echo "=== Converting to safetensors → $OUTPUT_DIR ==="
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(1 - $RATIO)")
    CKPT="$SAVE_DIR/${MODEL_PREFIX}_${SUFFIX}_${KEEP}.pt"
    ST_DIR="$OUTPUT_DIR/${MODEL_PREFIX}_${SUFFIX}_${KEEP}"
    [ ! -f "$CKPT" ] && continue
    if [ ! -f "$ST_DIR/model.safetensors" ]; then
        echo "  converting: $CKPT"
        python convert_pt_to_safetensors.py "$CKPT" --output_dir "$OUTPUT_DIR" \
            2>&1 | tee -a "logs/${MODEL_TAG}_${SUFFIX}_convert_st.log"
    else
        echo "  already converted: $ST_DIR/model.safetensors"
    fi
done

echo ""
echo "=== All done ==="
for RATIO in 0.2 0.3 0.4 0.5 0.6; do
    KEEP=$(python -c "print(1 - $RATIO)")
    ST_DIR="$OUTPUT_DIR/${MODEL_PREFIX}_${SUFFIX}_${KEEP}"
    [ -f "$ST_DIR/model.safetensors" ] && echo "  ✓ $ST_DIR" || echo "  ✗ MISSING: $ST_DIR"
done
