#!/bin/bash
# One-click SVD-LLM compression (V2 hetero + Basis Sharing) for all models.
# Runs sequentially: Llama-3.1-8B → Llama-3.1-8B-Instruct → Qwen3-8B → Qwen3-8B-Base
# Each model runs V2 first, then Basis Sharing.
#
# Usage:
#   bash run_compress_all.sh [HF_TOKEN]
#
# Each sub-script handles: compress → strip RoPE → safetensors.
# Already-done checkpoints are skipped automatically.

set -eo pipefail
cd "$(dirname "$0")"

HF_TOKEN="${1:-}"
TOKEN_ARG=""
[ -n "$HF_TOKEN" ] && TOKEN_ARG="$HF_TOKEN"

# ── Llama-3.1-8B ──────────────────────────────────────────────────────────────
echo "========================================"
echo "  Llama-3.1-8B  V2 hetero"
echo "========================================"
bash run_compress_llama31_8b_v2.sh $TOKEN_ARG

echo ""
echo "========================================"
echo "  Llama-3.1-8B  Basis Sharing (cross-layer)"
echo "========================================"
bash run_compress_llama31_8b_basissharing_cl.sh

# ── Llama-3.1-8B-Instruct ─────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Llama-3.1-8B-Instruct  V2 hetero"
echo "========================================"
bash run_compress_llama31_8b_instruct_v2.sh $TOKEN_ARG

echo ""
echo "========================================"
echo "  Llama-3.1-8B-Instruct  Basis Sharing (cross-layer)"
echo "========================================"
bash run_compress_llama31_8b_instruct_bs_cl.sh

# ── Qwen3-8B ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Qwen3-8B  V2 hetero"
echo "========================================"
bash run_compress_qwen3_8b_v2.sh

echo ""
echo "========================================"
echo "  Qwen3-8B  Basis Sharing"
echo "========================================"
bash run_compress_qwen3_8b_basissharing.sh

# ── Qwen3-8B-Base ─────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Qwen3-8B-Base  V2 hetero"
echo "========================================"
bash run_compress_qwen3_8b_instruct_v2.sh

echo ""
echo "========================================"
echo "  Qwen3-8B-Base  Basis Sharing"
echo "========================================"
bash run_compress_qwen3_8b_instruct_basissharing.sh

echo ""
echo "======================================== ALL DONE ========================================"
