#!/usr/bin/env bash
# Batch convert all Llama 3.1 8B .pt checkpoints to HF directories.
# Run from any directory; all paths are relative to this script's location.
#
# Usage:
#   bash tools/convert_llama31_8b_all.sh [--gpu 0] [--dtype bf16]
#
# Output dirs are created inside PT_DIR (same folder as the .pt files).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERT_SVDLLM="$SCRIPT_DIR/convert_svdllm_to_hf_dir.py"
CONVERT_ASVD="$SCRIPT_DIR/convert_asvd_to_hf_dir.py"

PT_DIR="$HOME/lowrankarena/llama31_download/llama31_8b"

# ── parse optional args ────────────────────────────────────────────────────────
GPU_ARG=""
DTYPE_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)   GPU_ARG="--gpu $2";   shift 2 ;;
        --dtype) DTYPE_ARG="--dtype $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── SVD-LLM checkpoints ────────────────────────────────────────────────────────
# Covers: whitening_only, whitening_then_update, basis_sharing, v2
for pt in "$PT_DIR"/meta_llama_Llama_3.1_8B_*.pt; do
    [[ -f "$pt" ]] || continue
    base=$(basename "$pt" .pt)
    # Extract tag (method) and ratio
    tag=$(echo "$base" | grep -oP '(whitening_only|whitening_then_update|basis_sharing|v2)' || true)
    ratio=$(echo "$base" | grep -oP '[0-9]+\.[0-9]+$' || true)
    [[ -z "$tag" || -z "$ratio" ]] && { echo "  [skip] cannot parse tag/ratio: $base"; continue; }
    out="$PT_DIR/hf_${tag}_${ratio}"
    if [[ -d "$out" ]]; then
        echo "[skip] already exists: $out"
        continue
    fi
    echo "==> SVD-LLM  $tag  ratio=$ratio"
    python "$CONVERT_SVDLLM" --input "$pt" --output "$out" $GPU_ARG $DTYPE_ARG
done

# ── ASVD checkpoints ──────────────────────────────────────────────────────────
for pt in "$PT_DIR"/Llama_3.1_8B_asvd_raw_*.pt; do
    [[ -f "$pt" ]] || continue
    ratio=$(basename "$pt" .pt | grep -oP '[0-9]+\.[0-9]+$' || true)
    [[ -z "$ratio" ]] && { echo "  [skip] cannot parse ratio: $(basename "$pt")"; continue; }
    out="$PT_DIR/hf_asvd_raw_${ratio}"
    if [[ -d "$out" ]]; then
        echo "[skip] already exists: $out"
        continue
    fi
    echo "==> ASVD  ratio=$ratio"
    python "$CONVERT_ASVD" --input "$pt" --output "$out" $DTYPE_ARG
done

echo ""
echo "All done. HF dirs:"
ls -d "$PT_DIR"/hf_* 2>/dev/null || echo "  (none found)"
