# LowRankArena Note

This directory contains the vendored upstream SVD-LLM baseline snapshot used by the `compress/` layer.

- It is not part of the main LowRankArena benchmark runtime.
- For benchmark-time eval, memory, and speed, use the top-level scripts under [`scripts/`](../../../scripts/README.md).
- For arena-facing artifact generation, use the wrapper entrypoint in [`compress/svd/svd_llm.py`](../svd_llm.py).
- For the vLLM compatibility path used by the benchmark runtime, see [`src/vllm/`](../../../src/vllm/README.md).

## Relevant Files

- `SVDLLM.py` and `SVDLLM_v2.py`: upstream low-rank decomposition logic.
- `svdllm_gen.py`: generation/export helper from the vendored baseline.
- `quant_llama.py`: upstream quantization-related helper.
