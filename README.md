# LowRankArena

LowRankArena is a benchmark repository for evaluating low-rank checkpoints, compression variants, and speed/quality tradeoffs. The main benchmark path uses a thin Python wrapper around `lm-eval-harness 0.4.11` for accuracy and `vLLM 0.18.1` for speed, while `compress/` remains the optional artifact-generation layer.

## Scope

- `old/` is treated as archived material and is not part of the new scaffold.
- Checkpoint metadata is tracked in `checkpoints/index.csv`.
- Rich per-checkpoint sidecar metadata can live under `checkpoints/manifests/`.
- The source of truth for hosted checkpoints is the gated Hugging Face repository: `Duke-CEI-SVD/LowRankArena`.
- `compress/` is optional. Main benchmark runs should load released checkpoints directly.
- Normalized benchmark outputs now share one project-owned top-level JSON schema across `eval`, `speed`, and `memory`.

## Environment

Recommended local environment on this machine:

```bash
conda activate lowrankarena
python -V
```

The current `requirements.txt` is aligned to the observed `lowrankarena` environment stack:

- Python `3.13.5`
- PyTorch `2.10.0+cu128`
- Transformers `4.57.x`
- vLLM `0.18.1`
- LM-Eval-Harness `0.4.11`

For FlashAttention-2, many systems still install more reliably with:

```bash
pip install flash-attn --no-build-isolation
```

## Layout

```text
lowrankarena/
├── README.md
├── pyproject.toml
├── requirements.txt
├── compress/
│   ├── README.md
│   ├── common.py
│   ├── save.py
│   ├── svd/
│   │   ├── asvd.py
│   │   ├── basis_sharing.py
│   │   ├── dobi_svd.py
│   │   ├── fwsvd.py
│   │   ├── svd.py
│   │   └── svd_llm.py
│   ├── prune/
│   │   ├── bonsai.py
│   │   ├── llm_pruner.py
│   │   ├── slicegpt.py
│   │   └── wanda_sp.py
│   └── quant/
│       ├── awq.py
│       ├── gptq.py
│       └── rtn.py
├── src/
│   ├── __init__.py
│   ├── arena.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── common.py
│   │   ├── llama/
│   │   │   ├── configuration_lowrank_llama.py
│   │   │   └── modeling_lowrank_llama.py
│   │   └── qwen/
│   │       ├── configuration_lowrank_qwen2.py
│   │       ├── modeling_lowrank_qwen2.py
│   │       ├── configuration_lowrank_qwen3.py
│   │       └── modeling_lowrank_qwen3.py
│   ├── load.py
│   ├── loader.py
│   ├── benchmarking.py
│   ├── lm_eval_runner.py
│   ├── memory_runner.py
│   ├── speed_runner.py
│   ├── scoring.py
│   ├── eval.py
│   ├── memory.py
│   ├── result_schema.py
│   ├── speed.py
│   ├── report.py
│   ├── registry.py
│   └── utils.py
├── benchmark/
│   ├── accuracy/
│   │   ├── mcq.yaml
│   │   ├── ppl.yaml
│   │   └── mmlu.yaml
│   ├── speed/
│   │   └── speed.yaml
│   └── main.yaml
├── scripts/
│   ├── run_eval.py
│   ├── run_memory.py
│   ├── run_speed.py
│   ├── run_all.py
│   ├── run_main.py
│   ├── run_compress.py
│   ├── make_table.py
│   └── add_checkpoint.py
├── checkpoints/
│   ├── index.csv
│   ├── vllm/
│   │   └── README.md
│   ├── manifests/
│   │   └── README.md
│   └── README.md
├── results/
│   ├── eval/
│   ├── memory/
│   ├── speed/
│   ├── tables/
│   └── figures/
└── tests/
    ├── test_load.py
    ├── test_eval.py
    └── test_manifest.py
```

## Workflow

1. Default reviewer path:
   - maintain released checkpoint metadata in `checkpoints/index.csv`
   - load checkpoints directly with `src/load.py` / `src/loader.py`
   - run accuracy suites through `src/lm_eval_runner.py`
   - run active-memory checks through `src/memory_runner.py`
   - run speed suites through `src/speed_runner.py`
   - use `scripts/run_eval.py`, `scripts/run_memory.py`, `scripts/run_speed.py`, and `scripts/run_main.py` as CLI entrypoints
2. Optional author/extender path:
   - generate a local artifact with `scripts/run_compress.py`
   - export a uniform manifest under `compress/artifacts/`
   - optionally register the artifact into `checkpoints/index.csv`
   - then run the normal eval, memory, and speed flow

## Programmatic API

The minimal public API is `Arena`. It wraps the existing registry, loader, eval, memory, speed, and reporting modules without introducing a second object model.

```python
from src import Arena

arena = Arena()

for row in arena.list(enabled_only=False):
    print(row["id"], row["method"])

print(arena.describe("llama31-8b-svdllm-0.6")["subpath"])
```

`Arena.register(...)` can add an in-memory checkpoint record or persist it to `checkpoints/index.csv`. `Arena.register_manifest(...)` imports a sidecar manifest from `checkpoints/manifests/` and writes the simplified runnable row into the registry when requested.

Low-rank execution is organized around shared family runtimes under `src/modeling/`. The intended contract is: export a method-specific checkpoint into the arena low-rank schema, then register it. The benchmark runtime is keyed by supported base families such as `llama` and `qwen`, not by per-checkpoint custom forward code. During export, the family-specific runtime files are copied from `src/modeling/<family>/` into the generated Hugging Face artifact.

`compress/` is intentionally not a second benchmarking framework. It only handles artifact generation.

In practice, `compress/svd/` and `compress/prune/` should be treated as method wrappers around heterogeneous baselines, while `compress/quant/` is kept as a more practical local path because quantization is much more likely to run inside the shared `lowrankarena` environment. See `compress/README.md` for the full rationale.

The benchmark configs are now separated by objective:

- `benchmark/accuracy/mcq.yaml` for exact lm-eval-harness `0.4.11` MCQ task IDs: `boolq`, `arc_easy`, `arc_challenge`, `winogrande`, `piqa`, `hellaswag`, `openbookqa`
- `benchmark/accuracy/ppl.yaml` for exact lm-eval-harness `0.4.11` rolling-loglikelihood task IDs: `wikitext`, `paloma_ptb`, `c4`
- `benchmark/accuracy/mmlu.yaml` for the official lm-eval-harness `0.4.11` group name: `mmlu`
- `scripts/run_memory.py` for Transformers-based active peak-memory measurement
- `benchmark/speed/speed.yaml` for vLLM offline inference speed
- `benchmark/main.yaml` as the aggregate entrypoint

Accuracy configs intentionally use the exact task names exposed by lm-eval-harness. In particular, `c4_stream` is not an lm-eval-harness `0.4.11` task ID in the current environment, so the benchmark config uses `c4` instead.

The accuracy runner calls the `lm-eval run ...` CLI rather than importing deep harness internals. The speed runner uses the `vllm.LLM` Python API directly.
The memory runner uses plain `transformers` generation plus `torch.cuda.max_memory_allocated()` so its peak measurements are not inflated by vLLM KV-cache reservation.

Two practical caveats from local smoke runs:

- `benchmark/accuracy/ppl.yaml` uses `paloma_ptb`, which requires access to the gated `allenai/paloma` dataset on Hugging Face.
- On shared GPU machines, `scripts/run_speed.py` may need a lower `--gpu-memory-utilization` value than the default benchmark config, and `--enforce-eager` can be useful for smoke runs.
- For custom long-context exports, `scripts/run_speed.py --max-model-len <N>` can be necessary to keep vLLM KV-cache requirements aligned with the actual benchmark prompt lengths.
- `scripts/run_memory.py` measures this process only, so other jobs on the same GPU can still cause OOM, but they do not pollute the reported `peak_allocated` value.

## Loader Example

```python
from src.load import load_checkpoint

loaded = load_checkpoint(
    "llama31-8b-svdllm-0.6",
    load_config=True,
)

print(loaded.record.subpath)
print(loaded.config.model_type)
```

The default `llama31-8b-svdllm-0.6` entry resolves to `llama31_8b/SVDLLMv1/hf_whitening_then_update_0.6`.

This alias is an inference-based default chosen because prior local benchmark artifacts referenced that export path. The manifest also keeps the exact `hf_whitening_only_0.6`, `hf_whitening_then_update_0.6`, and `hf_v2_0.6` entries separately.

## Compression Example

```bash
python scripts/run_compress.py \
  --family svd \
  --method asvd \
  --model meta-llama/Llama-3.1-8B \
  --ratio 0.5
```

This writes a planned local artifact under `compress/artifacts/` with:

- `manifest.json`
- `compression_log.json`
- `planned_command.sh` when the method has a known external command template

For more detail, see `compress/README.md`.

## Common Commands

```bash
python scripts/run_eval.py llama31-8b-baseline --suite accuracy/mcq
python scripts/run_memory.py llama-7b-svdllm-v1-update-0.5 --batch-size 1 --prompt-length 32 --generation-length 8
python scripts/run_speed.py llama-7b-svdllm-v1-update-0.5 --batch-size 1 --prompt-length 32 --generation-length 8 --repeat 1 --warmup 0
```

## Next Build Steps

- Expand checkpoint rows from model-family folders to exact exported variants.
- Add richer result aggregation and plotting in `src/report.py`.
- Add retry / resume handling for long-running benchmark jobs.
