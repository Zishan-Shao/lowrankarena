#!/bin/bash
#SBATCH -J llmpruner_l31_eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 2-00:00:00
#SBATCH -o /deac/csc/yangGrp/cuij/LLM/llm-pruner/logs/slurm_l31_eval_%j.out
#SBATCH -e /deac/csc/yangGrp/cuij/LLM/llm-pruner/logs/slurm_l31_eval_%j.err

set -euo pipefail
ratio="$1"
if [[ -z "$ratio" ]]; then
  echo "usage: $0 <ratio>" >&2
  exit 1
fi

source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
conda activate dobisvd
export PYTHONPATH=.
export HF_DATASETS_TRUST_REMOTE_CODE=1
cd /deac/csc/yangGrp/cuij/LLM/llm-pruner

BASE_MODEL="meta-llama/Llama-3.1-8B"
ckpt="/deac/csc/yangGrp/cuij/LLM/llm-pruner/prune_log/l31_8b_r${ratio}_prune/pytorch_model.bin"

python lm-evaluation-harness/main.py \
  --model hf-causal-experimental \
  --model_args checkpoint=${ckpt},config_pretrained=${BASE_MODEL} \
  --tasks openbookqa,arc_easy,arc_challenge,piqa,winogrande,hellaswag,boolq \
  --device cuda:0 \
  --output_path results/llama31_8b_r${ratio}_pruned_7task.json \
  --no_cache
