# `results/memory/`

This directory stores normalized memory results produced by the Transformers-based memory runner.

These files are intended to separate active per-process inference peaks from reserve-oriented backend behavior. They should be read alongside the speed results, not confused with vLLM KV-cache capacity logs.

## Expected Content

- One JSON file per checkpoint invocation.
- Shared top-level schema plus memory-specific peak numbers in `metrics` and breakdowns in `details`.
- Legacy `peak_memory__*.json` files may still appear from the older compatibility script path, but new runs should prefer the normalized `memory__*.json` naming.
