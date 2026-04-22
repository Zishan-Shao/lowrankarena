#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <retain_ratio> <state_dict_checkpoint> [extra lm-eval args...]" >&2
  exit 1
fi

ratio="$1"
ckpt="$2"
shift 2 || true

source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
conda activate dobisvd
export PYTHONPATH=.
export HF_DATASETS_TRUST_REMOTE_CODE=1
cd /deac/csc/yangGrp/cuij/LLM/llm-pruner

BASE_MODEL="/deac/csc/yangGrp/cuij/LLM/models/hf_models/huggyllama__llama-7b"
OUTPUT="/deac/csc/yangGrp/cuij/LLM/HAP-E/eval_results/llama1_7b/llama1_7b_keep_${ratio}_7task.json"
mkdir -p "$(dirname "$OUTPUT")"

python scripts/run_lm_eval_with_llama2_checkpoint.py \
  --model hf-causal-experimental \
  --model_args state_dict_checkpoint=${ckpt},config_pretrained=${BASE_MODEL} \
  --tasks openbookqa,arc_easy,arc_challenge,piqa,winogrande,hellaswag,boolq \
  --batch_size auto \
  --device cuda:0 \
  --output_path "$OUTPUT" \
  --no_cache \
  "$@"
