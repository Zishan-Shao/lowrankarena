#!/bin/bash
#SBATCH -J blockpruner_formal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_40
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH -o /deac/csc/yangGrp/cuij/LLM/BlockPruner/logs/slurm/%x_%j.out
#SBATCH -e /deac/csc/yangGrp/cuij/LLM/BlockPruner/logs/slurm/%x_%j.err

set -euo pipefail
MODEL_KEY="$1"
TARGETS_CSV="${2:-0.8,0.7,0.6,0.5,0.4}"
export PYTHONUNBUFFERED=1
bash /deac/csc/yangGrp/cuij/LLM/BlockPruner/scripts/run_formal_pipeline.sh "$MODEL_KEY" "$TARGETS_CSV"
