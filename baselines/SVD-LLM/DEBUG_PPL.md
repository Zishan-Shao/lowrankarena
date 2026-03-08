# FlashSVD PPL Debug Log

## 问题描述
SVD-LLM + FlashSVD MLP/attention kernel 开启时 PPL 从 42.63 → 207208。

## 完整 PPL 测试记录

### 原始 checkpoint（独立 gate_v/up_v），GPU 2

| attention kernel | MLP kernel | PPL |
|-----------------|------------|-----|
| fallback | fallback | **42.63** |
| fallback | FlashSVD v15（旧，V1顺序错） | 207208 |
| FlashSVD v16 | fallback | 42.63（bit-exact，见 test_e2e_logits.py）|

### shared-V checkpoint（gate_v = up_v，LSQ fix），GPU 2

| attention kernel | MLP kernel | PPL |
|-----------------|------------|-----|
| fallback | fallback | **89.07** |
| fallback | FlashSVD v15（V1顺序已修） | **89.07**（kernel 正确，质量损失来自 shared-V 近似）|
| FlashSVD v16 | FlashSVD v15（V1顺序已修） | 89.07 |

---

## 根因分析

### Bug 1：V1 gate/up 语义反转（已修复）

**位置**：`svd_llama.py` SVD_LlamaMLP.forward()

**原始代码（错误）**：
```python
V1 = torch.cat([self.up_u_proj.weight.t(), self.gate_u_proj.weight.t()], dim=1)
# kernel 计算: silu(up) * gate  ← 反了
```

**修复后（当前）**：
```python
V1 = torch.cat([self.gate_u_proj.weight.t(), self.up_u_proj.weight.t()], dim=1)
# kernel 计算: silu(gate) * up  ← 与 fallback 一致
```

**根因**：kernel 变量名 `Tu/Tv`（u=silu分支，v=线性分支）被误读为 `up/gate`，
导致 V1 拼接时 up 和 gate 位置互换。

---

### Bug 2：shared-V 约束导致质量损失

**问题**：`flashsvdswiglu_v15`（当前 kernel）只接受**单个 P** 输入：
```python
P = self.up_v_proj(x)   # 只做一次投影，gate 和 up 共用
V1 = cat([gate_u.T, up_u.T])
# gate 路径实际是: gate_u(up_v(x))，而不是 gate_u(gate_v(x))
```

原始 checkpoint 中 gate_v ≠ up_v（独立 SVD），gate 用错了 V → PPL 炸裂。

**当前 workaround（shared-V fix）**：
- 压缩时强制 gate_v = up_v，对 gate_u 做 LSQ 重投影
- PPL: 42.63（fallback）→ 89.07（kernel）
- 质量损失 = gate 信息被截断到 up_v 的行空间（R/D_in ≈ 36%）

**根本解决方案（待新 kernel）**：见下节。

---

## 新 kernel 替换指南

### 背景
`flashsvdswiglu_v15.py` 是旧版 kernel，组长确认有更新版本。
怀疑新版支持**独立双 P 输入**（gate 和 up 各自的 P），消除 shared-V 约束。

### 替换前确认新 kernel 的 API

**情形 A：新 kernel 支持双 P**
```python
# 预期新 API（假设）:
flashsvd_ffn_swiglu(Pg, Pu, V_gate, V_up, U2, V2, b1, b2)
# 或
flashsvd_ffn_swiglu(P_gate, P_up, ...)
```

**情形 B：新 kernel 仍是单 P，但有其他改进**
- 确认 silu 分支约定（`u` = gate/silu 还是 up？）
- 对应调整 V1 cat 顺序

### 替换步骤

**Step 1：替换 kernel 文件**
```
src/kernels/decoder/flashsvdswiglu_v15.py  ← 替换为新版
```
或新增文件（如 `flashsvdswiglu_v2.py`）并更新 import。

**Step 2：更新 svd_llama.py 的调用（根据新 API 调整）**

*如果新 kernel 支持双 P：*
```python
# svd_llama.py SVD_LlamaMLP.forward() 中：
if x.is_cuda:
    B, L, _ = x.shape
    D = self.up_u_proj.out_features
    Pg = self.gate_v_proj(x)          # gate 独立 P
    Pu = self.up_v_proj(x)            # up 独立 P
    Vg = self.gate_u_proj.weight.t()  # [R, D]
    Vu = self.up_u_proj.weight.t()    # [R, D]
    U2 = self.down_v_proj.weight.t()
    V2 = self.down_u_proj.weight.t()
    b1 = torch.zeros(2*D, ...)
    b2 = torch.zeros(V2.shape[1], ...)
    return new_flashsvd_ffn_swiglu(Pg, Pu, Vg, Vu, U2, V2, b1, b2)
```

*如果新 kernel 仍是单 P（但修复了其他问题）：*
```python
# 确保 V1 顺序正确（gate 在前）：
P  = self.up_v_proj(x)   # 或 gate_v_proj，取决于新 kernel 约定
V1 = torch.cat([self.gate_u_proj.weight.t(), self.up_u_proj.weight.t()], dim=1)
```

**Step 3：恢复原始 checkpoint（如果新 kernel 支持双 P）**
- 不需要 shared-V checkpoint，用原始 `whitening_only_0.5.pt` 即可
- 或重新跑 Step 1（不带 shared-V fix）生成干净的 checkpoint

**Step 4：验证**
```bash
# 目标 PPL：~42.63（与全 fallback 相同）
CUDA_VISIBLE_DEVICES=X python SVDLLM_flashsvd.py \
  --model jeffwan/llama-7b-hf \
  --step 4 \
  --model_path ./checkpoints/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
  --model_seq_len 2048 \
  --eval_batch_size 4
```

---

## 当前文件状态

| 文件 | 状态 |
|------|------|
| `svd_llama.py` | 两个 kernel 均开启；MLP 已切换双 P API |
| `SVDLLM_flashsvd.py` | shared-V fix 已写入（可切回原始 checkpoint）|
| `checkpoints/whitening_only_0.5.pt` | shared-V 版本（PPL fallback=89，kernel=89）|
| `src/kernels/decoder/flashsvdswiglu_v2.py` | **新版 kernel（双 P，支持独立 gate_v/up_v）** |
| `src/kernels/decoder/flashsvdswiglu_v15.py` | 旧版 kernel（已弃用，单 P）|
| `src/kernels/decoder/flashsvdropeattn_v16.py` | 已验证正确，无需替换 |
| `test_attn_kernel.py` | 合成数据诊断脚本 |
| `test_attn_real_model.py` | 真实 checkpoint 诊断脚本 |
| `test_e2e_logits.py` | 端到端 logit 对比脚本 |

## 新 kernel 集成（已完成）

`flashsvdswiglu_v2.py` 来自 `FlashSVD-v1.5 2/FlashSVD-v1.5/kernels/flashsvdswiglu.py`，
新增 `flashsvd_ffn_dual_split_token(PUp, PGate, GateU, UpU, DownV, DownU)` 支持独立双 P。

`svd_llama.py SVD_LlamaMLP.forward()` 已更新为：
```python
PGate = self.gate_v_proj(x)
PUp   = self.up_v_proj(x)
return flashsvd_ffn_dual_split_token(PUp, PGate, GateU, UpU, DownV, DownU)
```

**下一步**：用**原始** checkpoint（`whitening_only_0.5.pt` 无 shared-V fix 版本）跑 Step 4，目标 PPL ≈ 42.63。

---

## Attention kernel 验证结论（已完成，无需重做）

```
[PASS] logits finite=True  max_abs=0.000e+00  rel_fro=0.000e+00
CE loss fallback=13.1141  kernel=13.1141
```

`flashsvdropeattn_v16.py` bit-exact 正确，新 kernel 来了只需替换 MLP 部分。
