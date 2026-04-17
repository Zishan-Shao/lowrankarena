#!/bin/bash
# Batch-convert all llama2-7b checkpoints to HF dir format.
# Methods: SVD-LLM V1/V1update/BasisSharing + ASVD
# Keep ratios: 0.8, 0.7, 0.6, 0.5, 0.4  (compress ratios 0.2~0.6)
#
# Usage (run from lowrankarena/):
#   bash tools/convert_svdllm_llama2_7b_all.sh

set -eo pipefail
cd "$(dirname "$0")/.."

SVDLLM_CKPT_DIR="baselines/SVD-LLM/checkpoints/svdllm/llama2_7b"
ASVD_CKPT_DIR="baselines/ASVD/checkpoints/asvd/llama2_7b"
OUT_BASE="hf_ckpts/LowRankArena/llama2_7b"
PREFIX="meta_llama_Llama_2_7b_hf"
ASVD_PREFIX="Llama_2_7b_hf"
CONVERT_SVDLLM="tools/convert_svdllm_to_hf_dir.py"
CONVERT_ASVD="tools/convert_asvd_to_hf_dir.py"

convert_one() {
    local input="$1"
    local output="$2"
    local script="$3"
    if [ ! -f "$input" ]; then
        echo "  SKIP (not found): $input"
        return
    fi
    if [ -d "$output" ]; then
        echo "  SKIP (already exists): $output"
        return
    fi
    echo "  Converting: $input"
    python "$script" --input "$input" --output "$output"
}

for KEEP in 0.8 0.7 0.6 0.5 0.4; do
    echo "=== keep_ratio=$KEEP ==="

    # V1: whitening_only
    convert_one \
        "$SVDLLM_CKPT_DIR/${PREFIX}_whitening_only_${KEEP}.pt" \
        "$OUT_BASE/SVDLLMv1/hf_whitening_only_${KEEP}" \
        "$CONVERT_SVDLLM"

    # V1update: whitening_then_update (also under SVDLLMv1)
    convert_one \
        "$SVDLLM_CKPT_DIR/${PREFIX}_whitening_then_update_${KEEP}.pt" \
        "$OUT_BASE/SVDLLMv1/hf_whitening_then_update_${KEEP}" \
        "$CONVERT_SVDLLM"

    # V2: whitening_hetero
    convert_one \
        "$SVDLLM_CKPT_DIR/${PREFIX}_v2_${KEEP}.pt" \
        "$OUT_BASE/SVDLLMv2/hf_v2_${KEEP}" \
        "$CONVERT_SVDLLM"

    # Basis Sharing
    convert_one \
        "$SVDLLM_CKPT_DIR/${PREFIX}_basis_sharing_${KEEP}.pt" \
        "$OUT_BASE/Basis_sharing/hf_basis_sharing_${KEEP}" \
        "$CONVERT_SVDLLM"

    # ASVD
    convert_one \
        "$ASVD_CKPT_DIR/${ASVD_PREFIX}_asvd_raw_${KEEP}.pt" \
        "$OUT_BASE/ASVD/hf_asvd_raw_${KEEP}" \
        "$CONVERT_ASVD"
done

echo ""
echo "=== Done. Output dirs ==="
ls "$OUT_BASE" 2>/dev/null || echo "  (none created)"
