#!/usr/bin/env bash
set -euo pipefail

source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
conda activate dobisvd
export PYTHONPATH=.
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONUNBUFFERED=1

cd /deac/csc/yangGrp/cuij/LLM/llm-pruner

BASE_MODEL="/deac/csc/yangGrp/cuij/LLM/models/hf_models/huggyllama__llama-7b"

python lm-evaluation-harness/main.py \
  --model hf-causal \
  --model_args pretrained=${BASE_MODEL},local_files_only=True,trust_remote_code=True \
  --tasks openbookqa,arc_easy,arc_challenge,piqa,winogrande,hellaswag,boolq \
  --batch_size auto \
  --device cuda:0 \
  --output_path results/llama1_7b_baseline_7task.json \
  --no_cache
