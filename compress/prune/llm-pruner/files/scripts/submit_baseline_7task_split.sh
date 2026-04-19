#!/bin/bash
set -euo pipefail

BASE_L31="meta-llama/Llama-3.1-8B"
BASE_L1="huggyllama/llama-7b"
TASKS=(openbookqa arc_easy arc_challenge piqa winogrande hellaswag boolq)

for base in "$BASE_L31" "$BASE_L1"; do
  if [[ "$base" == "$BASE_L31" ]]; then
    tag="llama31_8b"
  else
    tag="llama1_7b"
  fi

  for task in "${TASKS[@]}"; do
    job_name="llmpruner_${tag}_${task}"
    out="/deac/csc/yangGrp/cuij/LLM/llm-pruner/logs/slurm_${tag}_${task}_%j.out"
    err="/deac/csc/yangGrp/cuij/LLM/llm-pruner/logs/slurm_${tag}_${task}_%j.err"
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
cd /deac/csc/yangGrp/cuij/LLM/llm-pruner
python lm-evaluation-harness/main.py \
  --model hf-causal \
  --model_args pretrained=${base} \
  --tasks ${task} \
  --device cuda:0 \
  --output_path results/${tag}_${task}_baseline_7task.json \
  --no_cache
EOF
  done

done
