# LowRankArena

**[🤗 Hugging Face: checkpoints, results, and audits](https://huggingface.co/Duke-CEI-SVD/LowRankArena)**

[Checkpoints](https://huggingface.co/Duke-CEI-SVD/LowRankArena/tree/main/checkpoints) ·
[Evaluation and inference results](https://huggingface.co/Duke-CEI-SVD/LowRankArena/tree/main/results) ·
[Audit artifacts](https://huggingface.co/Duke-CEI-SVD/LowRankArena/tree/main/audits)

LowRankArena is a standardized evaluation platform for low-rank and compression
checkpoints. It provides matched accuracy, memory, and inference measurements,
together with runnable code and a hosted checkpoint zoo.

The repository is organized around three benchmark paths:

- accuracy via `lm-eval-harness`
- active GPU memory measurement via plain `transformers`
- speed via `vLLM`

`compress/` is the optional artifact-generation layer. The normal benchmark path does not depend on re-running compression.

## Main Paper Results

The tables below are a copyable Markdown version of the main standardized
leaderboard. All methods use matched task versions, evaluation scripts, and
uniform-precision parameter keep ratios. `Dense FP` is the uncompressed
reference. WikiText-2 (`WT2`) and C4 report perplexity (lower is better); all
other metrics are higher-is-better.

`MCQ Avg.` is the unweighted mean over BoolQ, ARC-Easy, ARC-Challenge,
WinoGrande, PIQA, HellaSwag, and OpenBookQA. `Q. Ret.` is the arithmetic mean
of dense-normalized retention on WT2 PPL, C4 PPL, MCQ Avg., MathQA, and
MMLU-Math. Results should be compared within the shared LowRankArena protocol.

### Llama-1-7B

| Keep | Method | WT2 PPL ↓ | C4 PPL ↓ | BoolQ | ARC-E | ARC-C | WinoG. | PIQA | HellaS. | OBQA | MCQ Avg. ↑ | MathQA | MMLU-M | Q. Ret. ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | Dense FP | 5.67 | 7.20 | 0.737 | 0.726 | 0.445 | 0.693 | 0.786 | 0.749 | 0.408 | 0.649 | 0.261 | 0.278 | 1.000 |
| 80% | ASVD | 8.60 | 11.04 | 0.743 | 0.628 | 0.388 | 0.669 | 0.750 | 0.696 | 0.392 | 0.609 | 0.241 | 0.211 | 0.787 |
| 80% | SVD-LLM v1 | 7.88 | 16.42 | 0.633 | 0.481 | 0.317 | 0.584 | 0.681 | 0.477 | 0.332 | 0.501 | 0.224 | 0.291 | 0.767 |
| 80% | DoBi-SVD | 9.23 | 19.01 | 0.620 | 0.500 | 0.305 | 0.635 | 0.665 | 0.570 | 0.385 | 0.526 | 0.229 | 0.272 | 0.732 |
| 80% | Basis Sharing | 7.74 | 15.51 | 0.645 | 0.614 | 0.367 | 0.645 | 0.711 | 0.633 | 0.400 | 0.574 | 0.239 | 0.205 | 0.747 |
| 80% | MoDeGPT | 6.92 | 10.87 | 0.647 | 0.587 | 0.353 | 0.637 | 0.724 | 0.582 | 0.336 | 0.552 | 0.254 | 0.304 | 0.880 |
| 60% | ASVD | 3839.83 | 4268.56 | 0.450 | 0.274 | 0.244 | 0.503 | 0.528 | 0.264 | 0.238 | 0.357 | 0.198 | 0.272 | 0.458 |
| 60% | SVD-LLM v1 | 13.74 | 55.03 | 0.380 | 0.392 | 0.258 | 0.538 | 0.572 | 0.350 | 0.304 | 0.399 | 0.222 | 0.296 | 0.615 |
| 60% | DoBi-SVD | 15.24 | 48.37 | 0.433 | 0.389 | 0.268 | 0.594 | 0.580 | 0.403 | 0.312 | 0.426 | 0.213 | 0.225 | 0.560 |
| 60% | Basis Sharing | 12.42 | 41.13 | 0.503 | 0.472 | 0.287 | 0.586 | 0.608 | 0.452 | 0.344 | 0.465 | 0.209 | 0.268 | 0.622 |
| 60% | MoDeGPT | 12.43 | 23.40 | 0.613 | 0.474 | 0.335 | 0.645 | 0.644 | 0.521 | 0.334 | 0.509 | 0.238 | 0.275 | 0.690 |

### Llama-3.1-8B

| Keep | Method | WT2 PPL ↓ | C4 PPL ↓ | BoolQ | ARC-E | ARC-C | WinoG. | PIQA | HellaS. | OBQA | MCQ Avg. ↑ | MathQA | MMLU-M | Q. Ret. ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | Dense FP | 6.24 | 9.10 | 0.831 | 0.824 | 0.549 | 0.746 | 0.812 | 0.793 | 0.454 | 0.716 | 0.396 | 0.437 | 1.000 |
| 80% | ASVD | 2011.38 | 1281.96 | 0.382 | 0.285 | 0.226 | 0.512 | 0.536 | 0.285 | 0.244 | 0.353 | 0.201 | 0.223 | 0.304 |
| 80% | SVD-LLM v1 | 14.83 | 80.94 | 0.661 | 0.528 | 0.315 | 0.645 | 0.639 | 0.476 | 0.350 | 0.516 | 0.256 | 0.305 | 0.520 |
| 80% | DoBi-SVD | 556.59 | 1008.41 | 0.378 | 0.298 | 0.226 | 0.516 | 0.522 | 0.282 | 0.266 | 0.355 | 0.205 | 0.292 | 0.341 |
| 80% | Basis Sharing | 15.61 | 54.36 | 0.632 | 0.637 | 0.367 | 0.667 | 0.701 | 0.548 | 0.372 | 0.561 | 0.248 | 0.297 | 0.531 |
| 80% | MoDeGPT | 9.01 | 17.68 | 0.412 | 0.715 | 0.436 | 0.730 | 0.743 | 0.710 | 0.382 | 0.590 | 0.344 | 0.407 | 0.766 |
| 60% | ASVD | 22684.63 | 14186.23 | 0.405 | 0.254 | 0.257 | 0.491 | 0.507 | 0.260 | 0.284 | 0.351 | 0.192 | 0.286 | 0.326 |
| 60% | SVD-LLM v1 | 199.84 | 1187.78 | 0.378 | 0.295 | 0.246 | 0.533 | 0.515 | 0.283 | 0.268 | 0.360 | 0.205 | 0.257 | 0.329 |
| 60% | DoBi-SVD | 987.51 | 1529.38 | 0.378 | 0.271 | 0.251 | 0.481 | 0.511 | 0.265 | 0.288 | 0.349 | 0.203 | 0.268 | 0.325 |
| 60% | Basis Sharing | 82.96 | 461.21 | 0.380 | 0.409 | 0.241 | 0.562 | 0.568 | 0.325 | 0.284 | 0.396 | 0.205 | 0.211 | 0.330 |
| 60% | MoDeGPT | 24.50 | 51.82 | 0.622 | 0.460 | 0.312 | 0.672 | 0.629 | 0.516 | 0.316 | 0.504 | 0.241 | 0.213 | 0.446 |

### Qwen3-8B-Base

| Keep | Method | WT2 PPL ↓ | C4 PPL ↓ | BoolQ | ARC-E | ARC-C | WinoG. | PIQA | HellaS. | OBQA | MCQ Avg. ↑ | MathQA | MMLU-M | Q. Ret. ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | Dense FP | 7.00 | 11.78 | 0.830 | 0.800 | 0.570 | 0.727 | 0.793 | 0.787 | 0.420 | 0.704 | 0.542 | 0.729 | 1.000 |
| 80% | ASVD | 11.88 | 20.54 | 0.792 | 0.758 | 0.483 | 0.651 | 0.745 | 0.641 | 0.414 | 0.641 | 0.420 | 0.460 | 0.696 |
| 80% | SVD-LLM v1 | 11.08 | 33.90 | 0.688 | 0.649 | 0.424 | 0.669 | 0.712 | 0.616 | 0.408 | 0.595 | 0.330 | 0.355 | 0.584 |
| 80% | DoBi-SVD | 51.21 | 236.65 | 0.460 | 0.335 | 0.284 | 0.519 | 0.550 | 0.388 | 0.262 | 0.400 | 0.206 | 0.264 | 0.299 |
| 80% | Basis Sharing | 11.05 | 31.67 | 0.676 | 0.578 | 0.439 | 0.673 | 0.726 | 0.635 | 0.388 | 0.588 | 0.331 | 0.347 | 0.585 |
| 80% | MoDeGPT | 10.34 | 22.46 | 0.658 | 0.613 | 0.430 | 0.655 | 0.731 | 0.683 | 0.408 | 0.597 | 0.279 | 0.297 | 0.594 |
| 60% | ASVD | 1359.38 | 1484.57 | 0.502 | 0.292 | 0.241 | 0.499 | 0.544 | 0.281 | 0.270 | 0.375 | 0.212 | 0.234 | 0.252 |
| 60% | SVD-LLM v1 | 20.44 | 112.01 | 0.544 | 0.382 | 0.265 | 0.545 | 0.590 | 0.385 | 0.266 | 0.425 | 0.231 | 0.307 | 0.380 |
| 60% | DoBi-SVD | 107.63 | 610.21 | 0.511 | 0.290 | 0.253 | 0.511 | 0.523 | 0.310 | 0.306 | 0.386 | 0.203 | 0.302 | 0.284 |
| 60% | Basis Sharing | 18.59 | 95.39 | 0.630 | 0.457 | 0.278 | 0.580 | 0.620 | 0.423 | 0.298 | 0.469 | 0.253 | 0.302 | 0.409 |
| 60% | MoDeGPT | 18.40 | 55.66 | 0.509 | 0.414 | 0.293 | 0.569 | 0.625 | 0.460 | 0.328 | 0.457 | 0.228 | 0.246 | 0.400 |

The complete hosted result collection—including raw outputs, normalized
results, paper tables, serving measurements, and diagnostics—is available in
the [Hugging Face results directory](https://huggingface.co/Duke-CEI-SVD/LowRankArena/tree/main/results).

## Reproducible Audits

Small, release-facing audit reproductions live in [`audits/`](./audits/README.md).
They operate on immutable Hugging Face inputs and results; none of them
recompresses a model or regenerates a checkpoint.

- [Calibration rank stability](./audits/calibration_data_sensitivity/rank_stability/README.md):
  recompute the four-method math-retention ranking across three WikiText draws
  and one C4 draw.
- [Calibration-set source](./audits/calibration_data_sensitivity/calibration_source/README.md):
  verify source provenance, selection controls, tensor layout, row hashes, and
  SHA-256 checksums for the method-independent calibration inputs.
- [MoDeGPT BoolQ](./audits/inference_sensitivity/modegpt_boolq/README.md):
  reproduce the below-random and majority-baseline observations from the
  complete BoolQ result files.

The corresponding hosted artifacts are available in the
[Hugging Face audits directory](https://huggingface.co/Duke-CEI-SVD/LowRankArena/tree/main/audits).

## Scope

- Benchmark registry: [`checkpoints/index.csv`](./checkpoints/index.csv)
- Optional rich checkpoint metadata: [`checkpoints/manifests/`](./checkpoints/manifests/README.md)
- vLLM-compatible wrapper checkpoints: [`checkpoints/vllm/`](./checkpoints/vllm/README.md)
- Benchmark outputs: [`results/`](./results/README.md)
- Reusable runtime code: [`src/`](./src/README.md)
- User-facing entrypoints: [`scripts/`](./scripts/README.md)
- Declarative benchmark suites: [`benchmark/`](./benchmark/README.md)

The hosted artifact source of truth is
[`Duke-CEI-SVD/LowRankArena`](https://huggingface.co/Duke-CEI-SVD/LowRankArena)
on Hugging Face. The repository is publicly discoverable; downloading gated
artifacts may require accepting their access conditions and signing in.

## Environment

Recommended environment on this machine:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate lowrankarena
python -V
```

For the full validated stack including the speed path, use Python `3.13.x`.
`vllm==0.18.1` currently requires Python `<3.14`, so Python `3.14+` can still
install the base runtime but will skip the `vllm` speed dependency.

To refresh an existing environment in place:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda env update -f environment.yml --prune
conda activate lowrankarena
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
├── audits/
├── benchmark/
├── checkpoints/
├── compress/
├── results/
├── scripts/
├── src/
└── tests/
```

Key subtrees:

- [`audits/`](./audits/README.md): small, pinned reproductions for sensitivity and readiness claims
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
# accuracy
python scripts/run_eval.py llama31-8b-svdllm-v1-update-0.6 --suite accuracy/mcq --limit 1

# active memory
python scripts/run_memory.py \
  llama-7b-svdllm-v1-update-0.5 \
  --batch-size 1 \
  --prompt-length 32 \
  --generation-length 8

# vLLM speed
python scripts/run_speed.py \
  llama-7b-svdllm-v1-update-0.5 \
  --batch-size 1 \
  --prompt-length 32 \
  --generation-length 8 \
  --repeat 1 \
  --warmup 0

# quick smoke: load + eval(limit=1) + memory + speed
CUDA_VISIBLE_DEVICES=7 python demo.py llama31-8b-svdllm-v1-update-0.6 --device cuda:0

# more stable eval sanity check
CUDA_VISIBLE_DEVICES=7 python demo.py llama31-8b-svdllm-v1-update-0.6 --device cuda:0 --eval-limit 200
```

Verified low-overhead smoke commands for the currently supported checkpoint families:

```bash
# SVD-LLM v1
CUDA_VISIBLE_DEVICES=0 python demo.py llama-7b-svdllm-v1-update-0.5 \
  --skip-eval \
  --device cuda:0 \
  --memory-prompt-length 16 \
  --memory-generation-length 4 \
  --speed-prompt-length 16 \
  --speed-generation-length 4 \
  --speed-repeat 1 \
  --speed-warmup 0 \
  --speed-gpu-memory-utilization 0.2

# SVD-LLM v2 on Llama-3.1 / GQA
CUDA_VISIBLE_DEVICES=4 python demo.py llama31-8b-svdllm-v2-0.6 \
  --skip-eval \
  --device cuda:0 \
  --memory-prompt-length 16 \
  --memory-generation-length 4 \
  --speed-prompt-length 16 \
  --speed-generation-length 4 \
  --speed-repeat 1 \
  --speed-warmup 0 \
  --speed-gpu-memory-utilization 0.35

# ASVD on Llama-3.1 / GQA
CUDA_VISIBLE_DEVICES=5 python demo.py llama31-8b-asvd-0.4 \
  --skip-eval \
  --device cuda:0 \
  --memory-prompt-length 16 \
  --memory-generation-length 4 \
  --speed-prompt-length 16 \
  --speed-generation-length 4 \
  --speed-repeat 1 \
  --speed-warmup 0 \
  --speed-gpu-memory-utilization 0.25

# Basis sharing
CUDA_VISIBLE_DEVICES=3 python demo.py llama-7b-basis-sharing-0.5 \
  --skip-eval \
  --device cuda:0 \
  --memory-prompt-length 16 \
  --memory-generation-length 4 \
  --speed-prompt-length 16 \
  --speed-generation-length 4 \
  --speed-repeat 1 \
  --speed-warmup 0 \
  --speed-gpu-memory-utilization 0.2

CUDA_VISIBLE_DEVICES=4 python demo.py llama31-8b-basis-sharing-0.6 \
  --skip-eval \
  --device cuda:0 \
  --memory-prompt-length 16 \
  --memory-generation-length 4 \
  --speed-prompt-length 16 \
  --speed-generation-length 4 \
  --speed-repeat 1 \
  --speed-warmup 0 \
  --speed-gpu-memory-utilization 0.35

# DoBi
CUDA_VISIBLE_DEVICES=5 python demo.py llama31-8b-dobi-0.8 \
  --skip-eval \
  --device cuda:0 \
  --memory-prompt-length 16 \
  --memory-generation-length 4 \
  --speed-prompt-length 16 \
  --speed-generation-length 4 \
  --speed-repeat 1 \
  --speed-warmup 0 \
  --speed-gpu-memory-utilization 0.25

CUDA_VISIBLE_DEVICES=5 python demo.py llama-7b-dobi-0.8 \
  --skip-eval \
  --device cuda:0 \
  --memory-prompt-length 16 \
  --memory-generation-length 4 \
  --speed-prompt-length 16 \
  --speed-generation-length 4 \
  --speed-repeat 1 \
  --speed-warmup 0 \
  --speed-gpu-memory-utilization 0.2
```

## Notes

- `scripts/run_memory.py` reports this process's active GPU memory peak via `torch.cuda.max_memory_allocated()`. It is intentionally separate from vLLM's reserve-oriented KV-cache logs.
- `scripts/run_speed.py` uses the `src.vllm` adapter layer to prepare incompatible checkpoints before calling `vllm.LLM(...)`.
- Materialized vLLM wrapper checkpoints belong in [`checkpoints/vllm/`](./checkpoints/vllm/README.md), not under `src/vllm/`.
- `demo.py` defaults to `--eval-limit 1` so the eval step stays lightweight. That default is only for route checking. Use `--eval-limit 200` or run [`scripts/run_eval.py`](./scripts/run_eval.py) directly when you want a more stable accuracy sanity check.
