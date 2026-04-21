#!/bin/bash
# Run V2 heterogeneous compression for both Llama-3.1-8B and Llama-3.1-8B-Instruct.
# Runs base model first, then instruct, sharing the same HF token.
#
# Usage:
#   bash run_compress_llama31_all_v2.sh [HF_TOKEN]

set -eo pipefail
cd "$(dirname "$0")"

HF_TOKEN="${1:-}"

echo "========================================"
echo "  Llama-3.1-8B V2 compression"
echo "========================================"
bash run_compress_llama31_8b_v2.sh "$HF_TOKEN"

echo ""
echo "========================================"
echo "  Llama-3.1-8B-Instruct V2 compression"
echo "========================================"
bash run_compress_llama31_8b_instruct_v2.sh "$HF_TOKEN"

echo ""
echo "======================================== ALL DONE ========================================"
