# LowRankArena

LowRankArena is a scaffold-first benchmark repository for evaluating low-rank checkpoints, compression variants, and speed/quality tradeoffs. This version is intentionally lightweight: the goal is to establish a clean project layout, stable file formats, and clear extension points before wiring in the full execution logic.

## Scope

- `old/` is treated as archived material and is not part of the new scaffold.
- Checkpoint metadata is tracked in `checkpoints/index.csv`.
- The source of truth for hosted checkpoints is the gated Hugging Face repository: `Duke-CEI-SVD/LowRankArena`.
- Current Python modules return structured placeholder artifacts so the workflow can be refined incrementally.
- `compress/` is optional. Main benchmark runs should load released checkpoints directly.

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
│   ├── load.py
│   ├── eval.py
│   ├── speed.py
│   ├── report.py
│   ├── registry.py
│   └── utils.py
├── benchmark/
│   ├── main.yaml
│   ├── speed.yaml
│   ├── modern.yaml
│   ├── pruning.yaml
│   └── quant.yaml
├── scripts/
│   ├── run_eval.py
│   ├── run_speed.py
│   ├── run_all.py
│   ├── run_compress.py
│   ├── make_table.py
│   └── add_checkpoint.py
├── checkpoints/
│   ├── index.csv
│   └── README.md
├── results/
│   ├── eval/
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
   - load checkpoints directly with `src/load.py`
   - run eval, speed, and reporting from `scripts/`
2. Optional author/extender path:
   - generate a local artifact with `scripts/run_compress.py`
   - export a uniform manifest under `compress/artifacts/`
   - optionally register the artifact into `checkpoints/index.csv`
   - then run the normal eval and speed flow

`compress/` is intentionally not a second benchmarking framework. It only handles artifact generation.

In practice, `compress/svd/` and `compress/prune/` should be treated as method wrappers around heterogeneous baselines, while `compress/quant/` is kept as a more practical local path because quantization is much more likely to run inside the shared `lowrankarena` environment. See `compress/README.md` for the full rationale.

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

## Next Build Steps

- Replace stub metrics in `src/eval.py` and `src/speed.py` with real benchmark backends.
- Expand checkpoint rows from model-family folders to exact exported variants.
- Add richer result aggregation and plotting in `src/report.py`.
