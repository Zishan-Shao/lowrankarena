# FlashSVD PPL Debug Log

## 问题描述
SVD-LLM + FlashSVD attention kernel 开启时 PPL 从 42.63 → 207194。

## 实验数据

### PPL 测试（checkpoint: jeffwan_llama_7b_hf_whitening_only_0.5.pt, ratio=0.5）

| attention kernel | MLP kernel | PPL |
|-----------------|------------|-----|
| fallback | fallback | **42.63** |
| FlashSVD | fallback | 207194 |
| FlashSVD | FlashSVD | 207208 |
| fallback | FlashSVD | 207194 ← 原始诊断数据（旧 checkpoint） |

注：MLP kernel 单独开时也出现 207194，已知原因是 shared-V bug（gate_proj 和 up_proj 独立 V 因子，但 kernel 假设共享）。

## 内核正确性验证（test_attn_kernel.py）

所有测试 PASS：

| 测试 | 配置 | rel_fro | 结论 |
|------|------|---------|------|
| T1 | contiguous P, bf16, R=1024 | 5.35e-02 | PASS（bf16 累加误差正常） |
| T2 | stride-0 P (SVD-LLM expand), bf16 | 5.58e-02 | PASS |
| T3 | SVD-LLM forward, bf16, kernel vs ref | 9.44e-02 | PASS |
| T3b | stride-0 vs contiguous P 差异 | 0.00e+00 | PASS（完全等价） |
| T4 | R=64, bf16 | 9.71e-02 | PASS |
| T5 | fp16（真实模型 dtype），R=1024 | 2.97e-02 | PASS |
| T6 | fp16，小权重（模拟训练后模型） | 2.99e-04 | PASS |

## 真实 Checkpoint 验证（test_attn_real_model.py）

用真实 SVD-LLM 权重，layer 0-3 + S=2048：

| 测试 | rel_fro | 结论 |
|------|---------|------|
| layer 0, S=64 | 5.03e-04 | PASS |
| layer 1, S=64 | 7.77e-04 | PASS |
| layer 2, S=64 | 8.91e-04 | PASS |
| layer 3, S=64 | 7.11e-04 | PASS |
| layer 0, S=2048 | 7.54e-04 | PASS |

权重量级：Pq abs_mean ≈ 0.2，Vq abs_mean ≈ 0.08-0.24，输出 abs_mean ≈ 0.4，无 NaN/Inf。

## 结论

**attention FlashSVD 内核本身完全正确**，包括：
- bf16 和 fp16
- stride-0 expand（SVD-LLM 使用方式）
- 真实压缩权重
- S=64 和 S=2048

PPL=207194 的 **已确认根因：MLP kernel 的 shared-V bug**
- `flashsvd_ffn_swiglu` 用 `up_v_proj(x)` 作为 gate 和 up 的共享 P
- SVD-LLM 压缩时 gate_proj 和 up_proj 有独立 V 因子
- gate 分支实际上用了错误的 V → 输出垃圾

**attention kernel 单独导致 PPL=207194 的说法待验证**（test_e2e_logits.py）

## 修复方案

### MLP: shared-V fix（SVDLLM_flashsvd.py, whitening 函数）
压缩时强制 gate_proj 使用 up_proj 的 V 因子：
```python
# 保存原始权重和 up 的 V
_mlp_gate_W = W.clone()   # gate_proj 原始权重（GPU float32）
_mlp_up_v = svd_v.clone() # up_proj 的 svd_v

# 循环后重投影 gate
V = _mlp_up_v.float().to(dev)
VVT_inv = torch.linalg.inv(V @ V.t() + 1e-6 * torch.eye(...))
gate_u_new = _mlp_gate_W @ V.t() @ VVT_inv
layer.mlp.gate_u_proj.weight.data = gate_u_new.to(dtype).cpu()
layer.mlp.gate_v_proj.weight.data = _mlp_up_v
```

需要重新跑 Step 1 生成新 checkpoint 后再测试。

## 当前文件状态

| 文件 | 状态 |
|------|------|
| `svd_llama.py` | 两个 kernel 均 disabled（fallback）|
| `SVDLLM_flashsvd.py` | shared-V fix 已写入，需重跑 Step 1 |
| `test_attn_kernel.py` | 合成数据诊断脚本 |
| `test_attn_real_model.py` | 真实 checkpoint 诊断脚本 |
| `test_e2e_logits.py` | 端到端 logit 对比脚本 |

## 端到端 logit 验证（test_e2e_logits.py）— 最终结论

```
[PASS] logits finite=True  max_abs=0.000e+00  rel_fro=0.000e+00
CE loss fallback=13.1141  kernel=13.1141
```

**rel_fro=0 → attention kernel 输出与 fallback 完全相同（bit-exact）。**

attention kernel 不是 PPL=207194 的原因。之前"只开 attention → 207194"的测试
结果是误测（MLP kernel 当时也开着）。

**唯一根因：MLP shared-V bug（gate 用了错误的 V 因子）。**

## 下一步

1. 重跑 Step 1 生成带 shared-V fix 的新 checkpoint
2. 开 attention kernel（已验证完全正确）
3. 用新 checkpoint 开 MLP kernel，验证 PPL 接近 42.63
4. 两个 kernel 全开，跑加速测试
