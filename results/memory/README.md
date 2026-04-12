# `results/memory/`

This directory stores normalized memory results produced by the Transformers-based memory runner.

These files are intended to separate active per-process inference peaks from reserve-oriented backend behavior. They should be read alongside the speed results, not confused with vLLM KV-cache capacity logs.

## Expected Content

- One JSON file per checkpoint invocation.
- Stable top-level metadata for checkpoint identity, backend identity, runtime config, and peak-memory metrics.
