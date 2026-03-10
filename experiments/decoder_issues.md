# Decoder FlashSVD Issues

## ASVD + FlashSVD 兼容性问题

### 根本原因
ASVD 使用 per-layer 非均匀压缩（基于 sensitivity），导致与 FlashSVD kernel 的硬性约束冲突。

### FlashSVD kernel 要求
- q/k/v projections 必须**全部是 SVDLinear**（低秩分解）
- q/k/v 的 rank 必须**相等**（Rq == Rk == Rv）
- MLP gate/up 的 rank 必须**相等**（R_gate == R_up）
- Rank 需在合理范围内（**R ≲ 512**），否则 kernel overhead 超过收益

### ASVD 的实际行为（ratio=0.5, ppl sensitivity, LLaMA-7B）

| 模块 | 实际 Rank | FlashSVD 要求 | 结果 |
|------|-----------|--------------|------|
| q/k/v attention | R=819（均匀）| == 且 ≲512 | rank 过高，收益小 |
| gate_proj | 1194~1791（层间不均匀）| == up_proj | 部分层不等（37/64 需截断）|
| up_proj | 1194~1492（层间不均匀）| == gate_proj | 同上 |

- Layer 0：gate=1194，up=1492（不等）→ MLP kernel 禁用
- 大多数层 rank 在 1194~1791，远超 kernel 甜点范围

### SVD-LLM 无此问题的原因
- 全局统一 `param_ratio` 压缩所有层，q/k/v、gate/up 天然同 rank
- rank 由 param_ratio 控制，0.5 压缩比时 attention R≈1024（略高），MLP R≈1638
- 直接从压缩设计上保证 kernel 兼容性

---

## 实验结果汇总（LLaMA-7B, bf16, prompt=512, new_tokens=128, warmup=5, batch=1, L40S）

### SVD-LLM（作为参照）
| 模式 | ms/token | tok/s | speedup |
|------|---------|-------|---------|
| SVD baseline（FA2） | 30.5 | 33 | — |
| FlashSVD（FA2 + kernel） | 21.9 | 46 | **1.39x** |

### ASVD ratio=0.5（原始 checkpoint，R_attn=819）

| 模式 | ms/token | tok/s | speedup |
|------|---------|-------|---------|
| ASVD baseline（SDPA） | 32.8 | 30 | — |
| ASVD+FlashSVD | 31.4 | 32 | **1.05x** |

注：1.05x 几乎全部来自 **FA2 替换 SDPA**（KV cache 后端差异），reconstruct kernel 净贡献接近 0，MLP kernel 因 gate≠up rank 对部分层禁用，且所有层 rank 过高。

### ASVD + force_uniform_qkv（强制截断 attention rank，仅测速）

| target_rank | baseline ms | FlashSVD ms | speedup | 备注 |
|-------------|------------|-------------|---------|------|
| 819（原始） | 32.8 | 31.4 | 1.05x | MLP kernel 部分禁用 |
| 512 | ~33 | 29.8 | **1.19x** | MLP kernel 仍禁用 |
| 256 | 32.5 | 29.5 | 1.10x | FlashSVD decode floor ~29.5ms |
| 128 | 32.3 | 34.8 | 0.93x | R=dh，kernel overhead 超收益 |

FlashSVD decode 有约 **29.5ms 的下限**（FA2 读 KV cache 的内存带宽瓶颈），R 再小也无法突破。

### ASVD + force_uniform_mlp（强制对齐 gate/up rank，auto=1194）
| 模式 | ms/token | speedup | 备注 |
|------|---------|---------|------|
| baseline | 32.9 | — | |
| FlashSVD | 51.9 | **0.63x** | R=1194 下 kernel 远慢于 SVDLinear |

MLP kernel 在 R=1194 时 V1=[1194, 22016]，远超设计范围，严重退化。

---

## 根因总结

FlashSVD kernel 是为**高压缩率、低 rank、均匀压缩**场景设计的。ASVD 的 ppl sensitivity 分配策略与之相反：

- **保精度优先** → attention/MLP 均保留高 rank
- **非均匀分配** → gate≠up（导致 MLP kernel 禁用）
- **rank 远超甜点**（R_attn=819, R_mlp=1194~1791 vs 甜点 R≲512）

ASVD+FlashSVD 是 **negative result**：非均匀高压缩率场景下 FlashSVD kernel 无收益。

---

## 基准测试脚本现状（baselines/ASVD/bench_asvd_decode.py）

经本次调试，脚本已与 SVD-LLM 流程对齐：

| 特性 | 修复前 | 修复后 |
|------|-------|-------|
| 模型重载隔离 | 共享同一模型对象 | 每个 mode 独立 load + del + gc |
| Triton autotune 触发 | prefill 触发 MLP autotune（39s）| `L>4` guard，prefill fallback |
| 每 token 内存分配 | Vq/Vk/Vv 每次 alloc（643MB/token）| `__init__` 预计算，零分配 |
| SDPA 后端不稳定 | 随机切换 flash/mem_efficient/math | `--sdp_backend mem_efficient` 锁定 |
| MLP 静态权重 | V1/b1 每次 alloc | patch 时预计算，零分配 |
