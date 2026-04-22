#!/bin/bash
#SBATCH -J llmpruner_l31_ppl
#SBATCH -p gpu_small
#SBATCH --gres=gpu:H200_141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH -o /deac/csc/yangGrp/cuij/LLM/llm-pruner/logs/slurm_l31_ppl_%j.out
#SBATCH -e /deac/csc/yangGrp/cuij/LLM/llm-pruner/logs/slurm_l31_ppl_%j.err

set -euo pipefail

TARGET="${1:-}"
OUTPUT_JSON="${2:-}"
C4_MAX_EVAL_TOKENS="${3:-262144}"

if [[ -z "$TARGET" || -z "$OUTPUT_JSON" ]]; then
  echo "usage: sbatch $0 <baseline|0.8|0.7|0.6|0.5|0.4> <output_json> [c4_max_eval_tokens]" >&2
  exit 1
fi

source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
conda activate dobisvd

cd /deac/csc/yangGrp/cuij/LLM/llm-pruner
bash /deac/csc/yangGrp/cuij/LLM/llm-pruner/scripts/run_llama31_contiguous_ppl.sh \
  "$TARGET" \
  "$OUTPUT_JSON" \
  "$C4_MAX_EVAL_TOKENS"
