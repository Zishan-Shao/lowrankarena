# Decoder FlashSVD Issues

## ASVD + FlashSVD 兼容性问题

### 根本原因
ASVD 使用 per-layer 非均匀压缩（基于 sensitivity），导致与 FlashSVD kernel 的硬性约束冲突。

### FlashSVD kernel 要求
- q/k/v projections 必须**全部是 SVDLinear**（低秩分解）
- q/k/v 的 rank 必须**相等**（Rq == Rk == Rv）

### ASVD 的实际行为
- `stable_rank` sensitivity metric 认为 q/k 最敏感 → 分配 `param_ratio=1`（不压缩）
- q/k 保持 `nn.Linear`（full rank），只有 v 被压缩为 SVDLinear
- 结果：kernel 条件不满足，全部 fallback，decode **反而变慢**（0.86x）

### SVD-LLM 无此问题的原因
- 全局统一 `param_ratio` 压缩所有层，q/k/v 天然同 rank
- 直接从压缩设计上保证 kernel 兼容性

### 临时解决方案（仅用于测速）
通过 `--force_uniform_qkv --target_rank R` 强制对齐：

```bash
python baselines/ASVD/bench_asvd_decode.py \
    --checkpoint checkpoints/asvd/jeffwan_llama_7b_hf_asvd_ratio0.5.pt \
    --dtype bf16 --prompt_len 512 --new_tokens 128 \
    --warmup 5 --force_uniform_qkv --target_rank 512
```

- `target_rank=0`（默认）：自动取现有 SVDLinear 中的最小 rank
- `target_rank=512`：手动指定

### Rank 与加速比的关系（LLaMA-7B, ratio=0.5）

| target_rank | R/dh | decode speedup |
|-------------|------|----------------|
| 1024        | 8    | 0.83x（慢）    |
| 512         | 4    | 1.14x ✅       |

**结论**：kernel 对 R/dh 比值敏感。R=1024（R/dh=8）时 kernel overhead 超过收益；R=512（R/dh=4）时有效。

### 注意事项
- `force_uniform_qkv` 会重新压缩 v（从 1024→512），baseline 与 FlashSVD 的模型精度不同，对比**不完全公平**
- 正确做法：压缩阶段强制 q/k/v 同 rank（如 SVD-LLM 的做法），而非事后对齐
- ASVD+FlashSVD 作为 **negative result** 记录：非均匀压缩方案与当前 FlashSVD kernel 不兼容
