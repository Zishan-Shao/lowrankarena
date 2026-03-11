## flashsvdropeattn_short

This folder now keeps only legacy short-context benchmark harnesses.

### Active files

- `bench_decode_stack_compare.py`
  Single-layer decode-stack benchmark for the user-facing comparison:
  original SVD-Llama attention / original SwiGLU / FlashSVD attention / FlashSVDSwiGLU.
- `kernels/flashsvdropeattn_dense_decode.py`
  Shared-rank token reconstruct kernels and the prepacked GEMM backend used by the short-context dense-KV path.
- `kernels/flashsvd-archive/v1.5/flashsvdropeattn_short/flashsvdropeattn_v1.5_decode.py`
  Current low-rank fused decode kernel baseline.
- `kernels/flashsvd-archive/v1.6/flashsvdropeattn_short/flashsvdropeattn_v1.6_decode_opt.py`
  Newer low-rank decode variants (`v1` / `v2`) used by `decode_compare.py` and `svd_llama.py`.
- `legacy/flashsvdropeattn_baseline.py`
  Older unfused short-path baseline kept only for archive comparison.

### Main commands

Original-vs-optimized single-layer decode comparison:

```bash
python /home/zs89/FlashSVD/FlashSVD-v1.5/benchmark/legacy/archive/flashsvdropeattn_short/bench_decode_stack_compare.py \
  --llama llama2-7b \
  --ratio 0.5 \
  --B 1 \
  --Ls 256,1024,4096,8192 \
  --dtype bf16
```

This stack benchmark is intentionally aligned to the original SVD-Llama decode setting:

- single-token decode (`q_len=1`)
- dense KV cache
- original attention = low-rank reconstruct + RoPE + `torch.matmul/softmax`
- optimized attention = packed rank projection + token reconstruct + `flash_attn_with_kvcache`
- original MLP = low-rank SwiGLU
- optimized MLP = `flashsvd_ffn_swiglu`

By default the MLP benchmark assumes `gate_v_proj == up_v_proj`, which matches the converted FlashSVD checkpoints used by `SVDLLM.py` / `SVDLLM_flashsvd.py`.

### Legacy

Earlier exploratory scripts were moved to [`legacy/`](./legacy):

- `bench_flashsvd_decode_ab.py`
- `bench_flashsvd_decode_sweep.py`
- `compare.py`
- `flashsvdropeattn_baseline.py`
- `flashsvdropeattn_v1.py`
- `flashsvdropeattn_v1.5.py`

They are kept for reference, but they are no longer the primary entrypoints for short-context decode work.

The current attention route benchmark was moved to
[`benchmark/attn/decode_compare.py`](../../attn/decode_compare.py).
