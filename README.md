# LowRankArena

LowRankArena is a benchmark scaffold for low-rank and compression checkpoints.

The repository is organized around three benchmark paths:

- accuracy via `lm-eval-harness`
- active GPU memory measurement via plain `transformers`
- speed via `vLLM`

`compress/` is the optional artifact-generation layer. The normal benchmark path does not depend on re-running compression.

## Scope

- Benchmark registry: [`checkpoints/index.csv`](./checkpoints/index.csv)
- Optional rich checkpoint metadata: [`checkpoints/manifests/`](./checkpoints/manifests/README.md)
- vLLM-compatible wrapper checkpoints: [`checkpoints/vllm/`](./checkpoints/vllm/README.md)
- Benchmark outputs: [`results/`](./results/README.md)
- Reusable runtime code: [`src/`](./src/README.md)
- User-facing entrypoints: [`scripts/`](./scripts/README.md)
- Declarative benchmark suites: [`benchmark/`](./benchmark/README.md)

The current hosted checkpoint source of truth is the gated Hugging Face repository `Duke-CEI-SVD/LowRankArena`.

## Environment

Recommended environment on this machine:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lowrankarena
pip install -r requirements.txt
python -V
```

Observed stack in the active environment:

- Python `3.13.5`
- PyTorch `2.10.0+cu128`
- Transformers `4.57.6`
- vLLM `0.18.1`
- LM-Eval-Harness `0.4.11`
- `requirements.txt` now pins the validated benchmark/runtime stack used for the repo-owned flows

## Layout

```text
lowrankarena/
├── README.md
├── TODO.md
├── benchmark/
├── checkpoints/
├── compress/
├── results/
├── scripts/
├── src/
└── tests/
```

Key subtrees:

- [`benchmark/`](./benchmark/README.md): version-controlled suite definitions
- [`checkpoints/`](./checkpoints/README.md): runnable checkpoint registry and wrapper checkpoint storage
- [`compress/`](./compress/README.md): optional artifact-generation wrappers and vendored baselines
- [`results/`](./results/README.md): normalized benchmark outputs
- [`scripts/`](./scripts/README.md): CLI entrypoints
- [`src/`](./src/README.md): reusable runtime and adapters
- [`tests/`](./tests/README.md): lightweight regression checks

Root-level helper:

- [`demo.py`](./demo.py): small end-to-end smoke pass over checkpoint loading plus eval, memory, and speed.

## Recommended Workflow

1. Register or select a checkpoint from [`checkpoints/index.csv`](./checkpoints/index.csv).
2. Run accuracy with [`scripts/run_eval.py`](./scripts/run_eval.py).
3. Run memory with [`scripts/run_memory.py`](./scripts/run_memory.py).
4. Run speed with [`scripts/run_speed.py`](./scripts/run_speed.py).
5. Consume normalized outputs from [`results/`](./results/README.md).

Optional author path:

1. Build or plan an artifact through [`scripts/run_compress.py`](./scripts/run_compress.py).
2. Export it into a loadable checkpoint directory.
3. Register it in [`checkpoints/index.csv`](./checkpoints/index.csv) or a sidecar manifest.
4. Use the normal eval, memory, and speed flow.

## Programmatic API

The intended public entrypoint is [`Arena`](./src/arena.py).

```python
from src import Arena

arena = Arena()

for row in arena.list(enabled_only=False):
    print(row["id"], row["method"])

print(arena.describe("llama31-8b-svdllm-0.6")["subpath"])
```

`Arena` wraps the current registry, loader, eval, memory, speed, and reporting modules without introducing a second object model.

## Benchmark Surfaces

- [`benchmark/accuracy/`](./benchmark/accuracy/README.md): accuracy suites backed by `lm-eval-harness`
- [`benchmark/speed/`](./benchmark/speed/README.md): speed suites backed by `vLLM`
- memory currently runs through [`scripts/run_memory.py`](./scripts/run_memory.py) and [`src/memory_runner.py`](./src/memory_runner.py); it does not yet have a YAML suite tree

The benchmark outputs for `eval`, `memory`, and `speed` now share one project-owned top-level JSON schema. See [`src/result_schema.py`](./src/result_schema.py).

## Common Commands

```bash
python scripts/run_eval.py llama31-8b-svdllm-v1-update-0.6 --suite accuracy/mcq --limit 1

python scripts/run_memory.py \
  llama-7b-svdllm-v1-update-0.5 \
  --batch-size 1 \
  --prompt-length 32 \
  --generation-length 8

python scripts/run_speed.py \
  llama-7b-svdllm-v1-update-0.5 \
  --batch-size 1 \
  --prompt-length 32 \
  --generation-length 8 \
  --repeat 1 \
  --warmup 0

# quick smoke: eval uses --eval-limit 1
CUDA_VISIBLE_DEVICES=7 python demo.py \
  llama31-8b-svdllm-v1-update-0.6 \
  --device cuda:0

# slower but more stable eval sanity check
CUDA_VISIBLE_DEVICES=7 python demo.py \
  llama31-8b-svdllm-v1-update-0.6 \
  --device cuda:0 \
  --eval-limit 200
```

## Notes

- `scripts/run_memory.py` reports this process's active GPU memory peak via `torch.cuda.max_memory_allocated()`. It is intentionally separate from vLLM's reserve-oriented KV-cache logs.
- `scripts/run_speed.py` uses the `src.vllm` adapter layer to prepare incompatible checkpoints before calling `vllm.LLM(...)`.
- Materialized vLLM wrapper checkpoints belong in [`checkpoints/vllm/`](./checkpoints/vllm/README.md), not under `src/vllm/`.
- `demo.py` defaults to `--eval-limit 1` so the eval step stays lightweight. That default is only for route checking. Use `--eval-limit 200` or run [`scripts/run_eval.py`](./scripts/run_eval.py) directly when you want a more stable accuracy sanity check.
