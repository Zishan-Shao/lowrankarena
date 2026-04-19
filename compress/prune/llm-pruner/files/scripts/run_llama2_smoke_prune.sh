#!/bin/bash
#SBATCH -J llmpruner_l2_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 1-00:00:00
#SBATCH -o /deac/csc/yangGrp/cuij/LLM/llm-pruner/logs/slurm_llama2_smoke_%j.out
#SBATCH -e /deac/csc/yangGrp/cuij/LLM/llm-pruner/logs/slurm_llama2_smoke_%j.err

set -euo pipefail
source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
conda activate dobisvd
export PYTHONPATH=.
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONUNBUFFERED=1
cd /deac/csc/yangGrp/cuij/LLM/llm-pruner

python hf_prune.py \
  --pruning_ratio 0.4 \
  --block_wise \
  --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
  --block_attention_layer_start 4 --block_attention_layer_end 30 \
  --pruner_type taylor --taylor param_first \
  --test_after_train --test_before_train --save_model \
  --device cuda --eval_device cuda \
  --base_model meta-llama/Llama-2-7b-hf \
  --save_ckpt_log_name l2_7b_smoke_r40
