#!/usr/bin/env bash
# Batch convert all Llama-3.1-8B-Instruct .pt checkpoints to HF directories.
# Run from: lowrankarena/
#
# Usage:
#   bash tools/convert_llama31_8b_instruct_all.sh [--dtype bf16]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERT_SVDLLM="$SCRIPT_DIR/convert_svdllm_to_hf_dir.py"
CONVERT_ASVD="$SCRIPT_DIR/convert_asvd_to_hf_dir.py"

SVDLLM_PT="baselines/SVD-LLM/checkpoints/svdllm/llama31_8b_instruct"
ASVD_PT="baselines/ASVD/checkpoints/asvd/llama31_8b_instruct"
HF_BASE="hf_ckpts/LowRankArena/llama31_8b_instruct"

DTYPE_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dtype) DTYPE_ARG="--dtype $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── SVD-LLM ──────────────────────────────────────────────────────────────────
declare -A TAG_TO_METHOD=(
    [whitening_only]="SVDLLMv1"
    [whitening_then_update]="SVDLLMv1"
    [basis_sharing]="Basis_sharing"
    [v2]="SVDLLMv2"
)

for pt in "$SVDLLM_PT"/meta_llama_Llama_3.1_8B_Instruct_*.pt; do
    base=$(basename "$pt" .pt | sed 's/meta_llama_Llama_3\.1_8B_Instruct_//')
    keep=$(echo "$base" | grep -oP '[0-9]+\.[0-9]+$' || true)
    tag=$(echo "$base" | grep -oP '^(whitening_only|whitening_then_update|basis_sharing|v2)' || true)

    [[ -z "$tag" || -z "$keep" ]] && { echo "[skip] $base"; continue; }

    method="${TAG_TO_METHOD[$tag]:-$tag}"
    out="$HF_BASE/$method/hf_${tag}_${keep}"

    if [[ -f "$out/lowrank_config.json" ]]; then
        echo "[skip] already complete: $out"
        continue
    fi

    echo "==> $method  $tag  keep=$keep"
    python "$CONVERT_SVDLLM" --input "$pt" --output "$out" $DTYPE_ARG
done

# ── ASVD ─────────────────────────────────────────────────────────────────────
for pt in "$ASVD_PT"/Llama_3.1_8B_Instruct_asvd_raw_*.pt; do
    keep=$(basename "$pt" .pt | grep -oP '[0-9]+\.[0-9]+$' || true)
    [[ -z "$keep" ]] && { echo "[skip] $(basename $pt)"; continue; }

    out="$HF_BASE/ASVD/hf_asvd_raw_$keep"

    if [[ -f "$out/lowrank_config.json" ]]; then
        echo "[skip] already complete: $out"
        continue
    fi

    echo "==> ASVD  keep=$keep"
    python "$CONVERT_ASVD" --input "$pt" --output "$out" $DTYPE_ARG
done

echo ""
echo "=== All done ==="
