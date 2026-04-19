# llm-pruner

Base repository: https://github.com/horseee/llm-pruner.git
Base commit: `128a07d977f9b205d60ab14cfbc6a78f8a8e39d2`

This bundle contains local compression-related additions in a shareable format:
- `files/`: full modified files and newly added scripts, preserving repo-relative paths
- `patches/local_changes.patch`: diff against the listed base commit for tracked modified files
- `apply_to_repo.sh`: copies bundled files back into a fresh checkout

## Tracked Files Modified
- `hf_prune.py`
- `llama3.py`
- `lm-evaluation-harness/lm_eval/models/huggingface.py`

## New Files Added
- `scripts/build_llama31_mcq_c4_summary.py`
- `scripts/eval_contiguous_ppl.py`
- `scripts/run_baseline_llama1_7b.sh`
- `scripts/run_baseline_llama31_8b.sh`
- `scripts/run_hape_llama1_contiguous_ppl.sh`
- `scripts/run_hape_llama1_eval_bundle.sh`
- `scripts/run_hape_llama1_pruned_eval_alltasks.sh`
- `scripts/run_llama1_baseline_contiguous_ppl.sh`
- `scripts/run_llama1_baseline_eval_alltasks.sh`
- `scripts/run_llama1_contiguous_ppl.sh`
- `scripts/run_llama1_eval_batch_gpu.sh`
- `scripts/run_llama1_eval_batch_local.sh`
- `scripts/run_llama1_eval_bundle.sh`
- `scripts/run_llama1_prune_local_batch.sh`
- `scripts/run_llama1_prune_ratio.sh`
- `scripts/run_llama1_pruned_eval_alltasks.sh`
- `scripts/run_llama2_prune_ratio.sh`
- `scripts/run_llama2_pruned_eval_alltasks.sh`
- `scripts/run_llama2_smoke_prune.sh`
- `scripts/run_llama31_contiguous_ppl.sh`
- `scripts/run_llama31_prune_ratio.sh`
- `scripts/run_llama31_pruned_eval_alltasks.sh`
- `scripts/run_llama31_pruned_eval_r02.sh`
- `scripts/run_llama31_pruned_eval_r05.sh`
- `scripts/run_llama31_smoke_prune.sh`
- `scripts/run_lm_eval_with_llama2_checkpoint.py`
- `scripts/slurm_llama1_baseline_mcq.sbatch`
- `scripts/slurm_llama1_baseline_ppl.sbatch`
- `scripts/slurm_llama31_contiguous_ppl.sbatch`
- `scripts/submit_baseline_7task_split.sh`
- `scripts/submit_llama31_pruned_eval.sh`
- `scripts/update_llama1_section_in_summary.py`

## Notes
- This bundle intentionally excludes logs, results, caches, and __pycache__ files.
- Some scripts are compression entrypoints, while others are evaluation or summary helpers that support the same workflow.
