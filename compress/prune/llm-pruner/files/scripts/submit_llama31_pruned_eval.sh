#!/bin/bash
set -euo pipefail

BASE_MODEL="meta-llama/Llama-3.1-8B"
TASKS=(openbookqa arc_easy arc_challenge piqa winogrande hellaswag boolq)
RATIOS=(0.4 0.6 0.8)

for ratio in "${RATIOS[@]}"; do
  prune_dir="/deac/csc/yangGrp/cuij/LLM/llm-pruner/prune_log/l31_8b_r${ratio}_prune"
  ckpt="${prune_dir}/pytorch_model.bin"

  for task in "${TASKS[@]}"; do
    job_name="llmpruner_l31_r${ratio}_${task}"
    out="/deac/csc/yangGrp/cuij/LLM/llm-pruner/logs/slurm_l31_r${ratio}_${task}_%j.out"
    err="/deac/csc/yangGrp/cuij/LLM/llm-pruner/logs/slurm_l31_r${ratio}_${task}_%j.err"
    sbatch <<EOF
#!/bin/bash
#SBATCH -J ${job_name}
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 2-00:00:00
#SBATCH -o ${out}
#SBATCH -e ${err}

set -euo pipefail
source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
conda activate dobisvd
export PYTHONPATH=.
export HF_DATASETS_TRUST_REMOTE_CODE=1
cd /deac/csc/yangGrp/cuij/LLM/llm-pruner
python lm-evaluation-harness/main.py \
  --model hf-causal-experimental \
  --model_args checkpoint=${ckpt},config_pretrained=${BASE_MODEL} \
  --tasks ${task} \
  --device cuda:0 \
  --output_path results/llama31_8b_r${ratio}_${task}_pruned_7task.json \
  --no_cache
EOF
  done

done
