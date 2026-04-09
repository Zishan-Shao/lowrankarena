# LowRankArena

LowRankArena is a scaffold-first benchmark repository for evaluating low-rank checkpoints, compression variants, and speed/quality tradeoffs. This version is intentionally lightweight: the goal is to establish a clean project layout, stable file formats, and clear extension points before wiring in the full execution logic.

## Scope

- `old/` is treated as archived material and is not part of the new scaffold.
- Checkpoint metadata is tracked in `checkpoints/index.csv`.
- The source of truth for hosted checkpoints is the gated Hugging Face repository: `Duke-CEI-SVD/LowRankArena`.
- Current Python modules return structured placeholder artifacts so the workflow can be refined incrementally.

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

1. Maintain checkpoint metadata in `checkpoints/index.csv`.
2. Define benchmark slices in `benchmark/*.yaml`.
3. Use `src/load.py` to resolve or load a specific checkpoint from the Hugging Face repo.
4. Use `scripts/run_eval.py`, `scripts/run_speed.py`, or `scripts/run_all.py` to emit structured JSON placeholders into `results/`.
5. Convert result artifacts into Markdown tables with `scripts/make_table.py`.

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

## Next Build Steps

- Replace stub metrics in `src/eval.py` and `src/speed.py` with real benchmark backends.
- Expand checkpoint rows from model-family folders to exact exported variants.
- Add richer result aggregation and plotting in `src/report.py`.
