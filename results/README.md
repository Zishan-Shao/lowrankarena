# `results/`

This directory contains generated benchmark outputs.

Nothing here is a source-of-truth benchmark definition. The source of truth remains the benchmark configs, checkpoint manifest, and runner code. Result files are build artifacts that can be regenerated.

## Structure

- [`eval/`](./eval/README.md): normalized accuracy result JSON files.
- [`memory/`](./memory/README.md): normalized active-memory result JSON files.
- [`speed/`](./speed/README.md): normalized speed result JSON files.
- [`tables/`](./tables/README.md): derived tables for reports or paper drafting.
- [`figures/`](./figures/README.md): derived plots and visual summaries.
