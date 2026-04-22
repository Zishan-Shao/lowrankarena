#!/bin/bash
#SBATCH -J llmpruner_l1b_base_mcq
#SBATCH -p gpu_small
#SBATCH --gres=gpu:V100_32:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o /deac/csc/yangGrp/cuij/LLM/llm-pruner/logs/slurm_llama1_7b_baseline_mcq_%j.out
#SBATCH -e /deac/csc/yangGrp/cuij/LLM/llm-pruner/logs/slurm_llama1_7b_baseline_mcq_%j.err

set -euo pipefail
source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
conda activate dobisvd
cd /deac/csc/yangGrp/cuij/LLM/llm-pruner
bash /deac/csc/yangGrp/cuij/LLM/llm-pruner/scripts/run_llama1_baseline_eval_alltasks.sh
