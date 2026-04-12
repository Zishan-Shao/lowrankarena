# `results/speed/`

This directory stores normalized speed benchmark results produced by the vLLM runner.

Each result file captures both aggregate statistics and per-case measurements so downstream reporting can remain deterministic and auditable.

## Expected Content

- One JSON file per `suite x checkpoint`.
- Shared top-level schema plus aggregate throughput and latency summaries in `metrics`.
- Per-workload case measurements keyed by prompt length, generation length, and batch size in `details.cases`.
