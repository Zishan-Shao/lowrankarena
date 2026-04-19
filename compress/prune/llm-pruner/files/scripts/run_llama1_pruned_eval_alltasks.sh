#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <prune_ratio> [extra lm-eval args...]" >&2
  exit 1
fi

ratio="$1"
shift || true

source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
conda activate dobisvd
export PYTHONPATH=.
export HF_DATASETS_TRUST_REMOTE_CODE=1
cd /deac/csc/yangGrp/cuij/LLM/llm-pruner

BASE_MODEL="/deac/csc/yangGrp/cuij/LLM/models/hf_models/huggyllama__llama-7b"
ckpt="/deac/csc/yangGrp/cuij/LLM/llm-pruner/prune_log/l1_7b_r${ratio}_prune/pytorch_model.bin"

python scripts/run_lm_eval_with_llama2_checkpoint.py \
  --model hf-causal-experimental \
  --model_args checkpoint=${ckpt},config_pretrained=${BASE_MODEL} \
  --tasks openbookqa,arc_easy,arc_challenge,piqa,winogrande,hellaswag,boolq \
  --batch_size auto \
  --device cuda:0 \
  --output_path results/llama1_7b_r${ratio}_pruned_7task.json \
  --no_cache \
  "$@"
