#!/usr/bin/env bash
set -euo pipefail

ROOT="/deac/csc/yangGrp/cuij/LLM/BlockPruner"
mkdir -p "$ROOT/logs/slurm"

submit_one() {
  local model_key="$1"
  local job_name="$2"
  sbatch --parsable -J "$job_name" "$ROOT/scripts/slurm/blockpruner_formal.sbatch" "$model_key"
}

JOB1="$(submit_one llama1_7b bp_l1_7b)"
JOB2="$(submit_one llama31_8b bp_l31_8b)"

echo "llama1_7b $JOB1"
echo "llama31_8b $JOB2"
