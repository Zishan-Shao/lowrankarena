# Compression Code Bundle

This archive collects local compression-related code additions for three method repos:
- `llm-pruner`
- `slicegpt` (repo: `TransformerCompression`)
- `blockpruner`

Layout:
- `<method>/README.md`: base repo, commit, and included file list
- `<method>/files/`: repo-relative files to copy into a clean checkout
- `<method>/patches/local_changes.patch`: tracked-file diff against the recorded commit
- `<method>/apply_to_repo.sh`: helper to copy bundled files into the target repo

This bundle excludes logs, result JSONs, caches, checkpoints, and `__pycache__` artifacts.
