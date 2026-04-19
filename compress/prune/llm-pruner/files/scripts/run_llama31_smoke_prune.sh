#!/bin/bash
#SBATCH -J llmpruner_l31_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 1-00:00:00
#SBATCH -o /deac/csc/yangGrp/cuij/LLM/llm-pruner/logs/slurm_llama31_smoke_%j.out
#SBATCH -e /deac/csc/yangGrp/cuij/LLM/llm-pruner/logs/slurm_llama31_smoke_%j.err

set -euo pipefail
source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
conda activate dobisvd
export PYTHONPATH=.
export HF_DATASETS_TRUST_REMOTE_CODE=1
cd /deac/csc/yangGrp/cuij/LLM/llm-pruner

python llama3.py \
  --pruning_ratio 0.4 \
  --device cuda --eval_device cuda \
  --base_model meta-llama/Llama-3.1-8B \
  --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
  --block_attention_layer_start 4 --block_attention_layer_end 30 \
  --save_ckpt_log_name l31_8b_smoke_r40 \
  --pruner_type taylor --taylor param_first \
  --max_seq_len 2048 \
  --test_after_train --test_before_train --save_model
