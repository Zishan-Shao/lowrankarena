#!/bin/bash
#SBATCH -J llmpruner_l31b_baseline
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 2-00:00:00
#SBATCH -o /deac/csc/yangGrp/cuij/LLM/llm-pruner/logs/slurm_llama31_8b_baseline_%j.out
#SBATCH -e /deac/csc/yangGrp/cuij/LLM/llm-pruner/logs/slurm_llama31_8b_baseline_%j.err

set -euo pipefail
source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
conda activate dobisvd
export PYTHONPATH=.
cd /deac/csc/yangGrp/cuij/LLM/llm-pruner
python lm-evaluation-harness/main.py \
  --model hf-causal \
  --model_args pretrained=meta-llama/Llama-3.1-8B \
  --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
  --device cuda:0 \
  --output_path results/llama31_8b_baseline_7task.json \
  --no_cache
