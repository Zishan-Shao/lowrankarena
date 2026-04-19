#!/bin/bash
#SBATCH -J llmpruner_l2_prune
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 3-00:00:00
#SBATCH -o /deac/csc/yangGrp/cuij/LLM/llm-pruner/logs/slurm_llama2_prune_%j.out
#SBATCH -e /deac/csc/yangGrp/cuij/LLM/llm-pruner/logs/slurm_llama2_prune_%j.err

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

save_tag="l2_7b_r${ratio}_prune"
python hf_prune.py \
  --pruning_ratio ${ratio} \
  --block_wise \
  --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
  --block_attention_layer_start 4 --block_attention_layer_end 30 \
  --pruner_type taylor --taylor param_first \
  --test_after_train --test_before_train --save_model \
  --device cpu --eval_device cuda \
  --base_model meta-llama/Llama-2-7b-hf \
  --save_ckpt_log_name ${save_tag}
