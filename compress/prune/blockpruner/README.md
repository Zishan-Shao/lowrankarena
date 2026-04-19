# blockpruner

Base repository: https://github.com/MrGGLS/BlockPruner.git
Base commit: `821070741e7b45683e54ed986e6bf29f66a88d5d`

This bundle contains local compression-related additions in a shareable format:
- `files/`: full modified files and newly added scripts, preserving repo-relative paths
- `patches/local_changes.patch`: diff against the listed base commit for tracked modified files
- `apply_to_repo.sh`: copies bundled files back into a fresh checkout

## Tracked Files Modified
- `block_search.py`
- `eval.py`
- `utils.py`

## New Files Added
- `scripts/calibrate_keep_ratio.py`
- `scripts/eval_standardized.py`
- `scripts/run_formal_pipeline.sh`
- `scripts/slurm/blockpruner_formal.sbatch`
- `scripts/submit_formal_jobs.sh`
- `scripts/update_summary.py`

## Notes
- This bundle intentionally excludes logs, results, caches, and __pycache__ files.
- Some scripts are compression entrypoints, while others are evaluation or summary helpers that support the same workflow.
