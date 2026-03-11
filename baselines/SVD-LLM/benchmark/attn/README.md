# Attention Benchmarks

This folder holds current attention-focused benchmarks that are worth exposing
next to `benchmark/decode/` and `benchmark/mlp/`.

Current entrypoints:

- `decode_compare.py`
  Single-token decode attention route comparison:
  dense KV + FA2, dense KV + token reconstruct + FA2 KVCache, and legacy
  low-rank decode kernels.

Example:

```bash
python /home/zs89/FlashSVD/FlashSVD-v1.5/benchmark/attn/decode_compare.py \
  --llama llama2-7b \
  --target-param-ratio 0.5 \
  --rank-formula global \
  --factor-layout shared \
  --B 1 \
  --Ls 256,512,1024,2048,4096,8192 \
  --dtype bf16 \
  --dense-backend fa2
```
