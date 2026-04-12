# `compress/artifacts/`

This directory stores locally generated compression artifacts and planning outputs.

Files written here are generated products, not hand-maintained source files. A complete artifact is expected to include a manifest and enough metadata for the normal benchmark path to load it later.

## Typical Contents

- `manifest.json`: normalized artifact metadata.
- `compression_log.json`: method-specific execution record.
- `planned_command.sh`: optional command snapshot for externally executed methods.

Only artifacts that are actually loadable should be registered into [`checkpoints/index.csv`](../../checkpoints/index.csv) or a richer sidecar manifest under [`checkpoints/manifests/`](../../checkpoints/manifests/README.md).
