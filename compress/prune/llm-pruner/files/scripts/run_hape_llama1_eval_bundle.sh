#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <retain_ratio> <state_dict_checkpoint>" >&2
  exit 1
fi

RETAIN="$1"
CKPT="$2"

ROOT="/deac/csc/yangGrp/cuij/LLM/llm-pruner"
SUMMARY_UPDATER="$ROOT/scripts/update_llama1_section_in_summary.py"

source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
conda activate dobisvd
cd "$ROOT"

bash "$ROOT/scripts/run_hape_llama1_pruned_eval_alltasks.sh" "$RETAIN" "$CKPT"
bash "$ROOT/scripts/run_hape_llama1_contiguous_ppl.sh" "$RETAIN" "$CKPT"
python "$SUMMARY_UPDATER"
