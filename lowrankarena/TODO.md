# TODO

This file tracks the shortest path from the current LowRankArena state to a stable `v1.0`.

## P0: Must Finish First

- [ ] Promote memory measurement into a first-class runner
  Target:
  - add `src/memory_runner.py`
  - add `src/memory.py`
  - add `scripts/run_memory.py`
  Goal:
  - memory should be a peer of `eval` and `speed`, not a one-off script

- [ ] Unify result schemas across `eval`, `speed`, and `memory`
  Target fields:
  - `checkpoint`
  - `backend`
  - `config`
  - `metrics`
  - `artifacts`
  - `timestamp`
  Goal:
  - downstream reporting should not need per-runner special cases

- [ ] Freeze the low-rank adapter interface
  Current anchor:
  - `src/vllm/vllm_adapter.py`
  Goal:
  - `src/speed_runner.py` should only consume a prepared model descriptor
  - checkpoint-specific logic should stay out of the runner

- [ ] Add minimal regression tests for the SVD-LLM path
  Must cover:
  - `load_checkpoint(...)`
  - `prepare_model_for_vllm(...)`
  - wrapper materialization is re-entrant
  - loaded parameters remain factorized (`u_proj` / `v_proj`) instead of dense `*.weight`

- [ ] Clean up default terminal output
  Goal:
  - default mode shows stage progress and final summaries only
  - verbose mode exposes raw backend logs when needed

- [ ] Write one top-level workflow doc
  Goal:
  - a new user should be able to add a checkpoint and run `eval`, `speed`, and `memory` from one page

## P1: Strongly Recommended

- [ ] Integrate memory into the benchmark suite system
  Target:
  - support something like `benchmark/memory/*.yaml`

- [ ] Add a unified summary table flow
  Goal:
  - one command should aggregate accuracy, speed, and memory outputs

- [ ] Expand checkpoint metadata in `checkpoints/index.csv`
  Useful fields:
  - `is_low_rank`
  - `weight_format`
  - `kv_format`
  - `compression_ratio`

- [ ] Standardize memory terminology in outputs
  Always distinguish:
  - weights memory
  - live KV memory
  - reserved KV memory
  - total active peak

- [ ] Add minimal CI for non-GPU checks
  Scope:
  - formatting
  - linting
  - lightweight unit tests

## P2: After The Core Is Stable

- [ ] Generalize adapters beyond SVD-LLM
- [ ] Explore native vLLM low-rank execution instead of the Transformers backend
- [ ] Build a stronger reporting / leaderboard layer
- [ ] Expand consistent support across more model families and compression methods

## v1.0 Exit Criteria

- [ ] A new user can follow the docs and run `eval`, `speed`, and `memory` without reading source code
- [ ] All three runner types emit normalized JSON with a shared top-level structure
- [ ] Speed and memory outputs clearly separate true usage from reserve/capacity numbers
- [ ] The SVD-LLM low-rank path is covered by automated regression tests
- [ ] The default CLI experience is clean enough to track progress without backend log spam

## Suggested Implementation Order

1. `run_memory.py` and `src/memory_runner.py`
2. Result schema unification
3. Adapter interface cleanup
4. Regression tests
5. Terminal/log cleanup
6. Top-level workflow documentation
