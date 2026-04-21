#!/bin/bash
# One-click SVD-LLM V2 heterogeneous compression for all models.
# Runs sequentially: Llama-3.1-8B → Llama-3.1-8B-Instruct → Qwen3-8B → Qwen3-8B-Instruct
#
# Usage:
#   bash run_compress_all_v2.sh [HF_TOKEN]
#
# Each sub-script handles: compress → strip RoPE → safetensors conversion.
# Already-done checkpoints are skipped automatically.

set -eo pipefail
cd "$(dirname "$0")"

HF_TOKEN="${1:-}"
TOKEN_ARG=""
[ -n "$HF_TOKEN" ] && TOKEN_ARG="$HF_TOKEN"

echo "========================================"
echo "  Llama-3.1-8B  V2 hetero"
echo "========================================"
bash run_compress_llama31_8b_v2.sh $TOKEN_ARG

echo ""
echo "========================================"
echo "  Llama-3.1-8B-Instruct  V2 hetero"
echo "========================================"
bash run_compress_llama31_8b_instruct_v2.sh $TOKEN_ARG

echo ""
echo "========================================"
echo "  Qwen3-8B  V2 hetero"
echo "========================================"
bash run_compress_qwen3_8b_v2.sh

echo ""
echo "========================================"
echo "  Qwen3-8B-Instruct  V2 hetero"
echo "========================================"
bash run_compress_qwen3_8b_instruct_v2.sh

echo ""
echo "======================================== ALL DONE ========================================"
