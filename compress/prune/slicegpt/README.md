# slicegpt

Base repository: https://github.com/microsoft/TransformerCompression
Base commit: `6b12cdee6ad51791d7c776b3a046bc408b9e77e9`

This bundle contains local compression-related additions in a shareable format:
- `files/`: full modified files and newly added scripts, preserving repo-relative paths
- `patches/local_changes.patch`: diff against the listed base commit for tracked modified files
- `apply_to_repo.sh`: copies bundled files back into a fresh checkout

## Tracked Files Modified
- `experiments/run_lm_eval.py`
- `experiments/run_slicegpt.py`
- `src/slicegpt/adapters/llama_adapter.py`
- `src/slicegpt/data_utils.py`
- `src/slicegpt/hf_utils.py`

## New Files Added
- `scripts/local/run_slicegpt_l1_7b_local_gpu.sh`
- `scripts/local/run_slicegpt_l1_c4_ppl_batch.sh`
- `scripts/local/run_slicegpt_l1_c4_ppl_keep.sh`
- `scripts/slurm/run_slicegpt_c4_ppl_batch.sh`
- `scripts/slurm/run_slicegpt_ratio.sh`
- `scripts/slurm/submit_l1_7b_batch.sh`
- `scripts/slurm/submit_l31_8b_batch.sh`

## Notes
- This bundle intentionally excludes logs, results, caches, and __pycache__ files.
- Some scripts are compression entrypoints, while others are evaluation or summary helpers that support the same workflow.
