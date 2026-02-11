# SVD-LLM v1 vs v2 实测对比

## 测试配置

**模型**: `textattack/bert-base-uncased-sst-2`
**任务**: SST-2 (Sentiment Classification)
**数据集**: GLUE SST-2 validation (872 samples, 28 batches)
**配置**:
- Batch size: 32
- Sequence length: 256
- Ratio: 0.5
- Device: CUDA

---

## Rank 配置

两个版本使用**相同的 rank 公式** (v1):

```python
def rank_from_ratio(m, n, ratio):
    return int(m * n * ratio / (m + n))

# BERT-base (dm=768, dff=3072, dh=64)
RANK_ATTN = rank_from_ratio(768, 64, 0.5)   = 29  # per-head Q/K/V
RANK_FF   = rank_from_ratio(768, 3072, 0.5) = 307 # FFN intermediate
RANK_WO   = rank_from_ratio(768, 768, 0.5)  = 192 # Attention output
```

**注意**: 当前两个版本都使用 v1 公式，v2 的差异不在 rank 计算，而在后续优化步骤。

---

## 实现差异

### v1: DRONE Only

**流程**:
1. 使用 Whiten/DRONE 方法进行 SVD 分解
   - 收集输入协方差矩阵 C = E[x x^T]
   - Cholesky 分解: C = L L^T
   - 在白化空间中 SVD: W_scale = L^T W
   - 最小化 ||X^T(W - UV)||_F

2. 直接使用分解后的 U, V 矩阵

**代码**: `src/encoders/BERTWhiting/profile_svdllm.py` (或 `profile_svdllm_v1.py`)

### v2: DRONE + Local Update

**流程**:
1. 使用 Whiten/DRONE 方法进行初始 SVD 分解（同 v1）

2. **额外的 Local Update 步骤** (FFN only):
   - 保留原始 dense 模型作为 teacher
   - 固定 V1, V2 (FFN 的右矩阵)
   - 通过 teacher-student IO pairs 更新 U1, U2 (左矩阵)
   - 使用最小二乘法求解: Y ≈ X @ U @ V (V 固定)

**关键代码** (`profile_svdllm_v2.py:390-510`):
```python
def svdllm_v2_local_update_ffn_only(student, teacher, loader, ...):
    # Fix V1, V2 in student
    V1 = blk.V1.detach()
    V2 = blk.V2.detach()

    # Collect teacher IO pairs
    # intermediate.dense: X1 -> Y1
    # output.dense:       X2 -> Y2

    # Solve for U1, U2
    U1_new = _solve_U_fixed_V(X1, Y1, V1, ridge)
    U2_new = _solve_U_fixed_V(X2, Y2, V2, ridge)

    # Update student
    blk.U1.data.copy_(U1_new)
    blk.U2.data.copy_(U2_new)
```

**目标**: 通过 knowledge distillation 让压缩模型更接近原始模型的输出

---

## 性能对比

### v1 结果

```
RANK_ATTN: 29  RANK_FF: 307  RANK_WO: 192

Accuracy      : 0.8817 (88.17%)
Latency       : 307.8 ms/batch
Peak Memory   : 714.9 MB
Model Size    : 254.8 MiB
  - Dense     : 93.2 MiB
  - Low-rank  : 161.6 MiB
```

### v2 结果

```
RANK_ATTN: 29  RANK_FF: 307  RANK_WO: 192

Accuracy      : 0.8002 (80.02%)  ⚠️ -8.15% vs v1
Latency       : 296.4 ms/batch   ✅ -3.7% faster
Peak Memory   : 1147.1 MB        ⚠️ +60.5% higher
Model Size    : 254.8 MiB        ✅ Same
  - Dense     : 93.2 MiB
  - Low-rank  : 161.6 MiB

Extra steps:
- "Running SVD-LLM v2 local update (FFN only)"
- Updated FFN U matrices for all 12 layers
```

### 对比 Dense Baseline

```
Dense (from eval_encoder):
Accuracy      : 0.9263 (92.63%)
Latency       : 65.01 ms/batch
Peak Memory   : 360.2 MB
Model Size    : ~440 MiB
```

---

## 关键发现

### 1. v2 准确率显著下降 ⚠️

- **v1**: 88.17% (相比 dense 下降 4.46%)
- **v2**: 80.02% (相比 dense 下降 12.61%)
- **v2 vs v1**: 下降 8.15%

**可能原因**:
1. **过拟合 calibration data**: local update 只用了 4 个 batch
2. **方法不适合 encoder**: SVD-LLM v2 可能针对 decoder LLM 设计
3. **实现问题**: 可能存在 bug 或参数设置不当
4. **数据分布差异**: calibration 和 validation 数据分布不同

### 2. v2 速度略快 (+3.7%)

- **v1**: 307.8 ms/batch
- **v2**: 296.4 ms/batch
- **提升**: 11.4 ms (-3.7%)

**分析**:
- 速度提升很小，在误差范围内
- 可能是因为 U 矩阵更新后的数值特性更好
- 但这点速度提升不值得损失 8% 的准确率

### 3. v2 内存显著增加 (+60.5%)

- **v1**: 714.9 MB
- **v2**: 1147.1 MB
- **增加**: 432.2 MB (+60.5%)

**可能原因**:
1. **teacher 模型**: v2 需要额外保存原始 dense 模型
2. **IO pairs 收集**: 中间需要存储大量的输入输出对
3. **中间计算**: local update 过程中的临时 tensor

### 4. 模型大小相同

- 两者都使用相同的 rank 配置
- 最终的参数量完全一致
- 差异仅在训练/优化过程

---

## v1 vs v2 使用建议

### 推荐使用 v1 ✅

**理由**:
1. ✅ **准确率高 8%** (88.17% vs 80.02%)
2. ✅ **内存占用低 40%** (714.9 MB vs 1147.1 MB)
3. ✅ **实现简单** (无需额外的 local update)
4. ✅ **训练快** (无需遍历数据集更新 U)

**适用场景**:
- 所有 BERT encoder 模型压缩
- 需要在准确率和压缩率之间平衡
- 资源受限环境

### v2 可能的使用场景 🤔

**当前不推荐**，因为准确率损失过大。

**潜在改进方向**:
1. 增加 calibration batch 数量 (4 → 16+)
2. 调整 ridge 参数 (1e-6 → 更大)
3. 对 Attention 层也应用 local update（当前只有 FFN）
4. 使用 validation set 的一部分进行 local update
5. 多轮迭代 local update

**如果改进后准确率提升**，可能适用于:
- 对速度敏感的场景（牺牲准确率换 3-4% 速度）
- 有大量 unlabeled data 可用于 local update
- 可以接受较低准确率的非关键任务

---

## v1 vs v2 vs Dense 三方对比

| 指标 | Dense | v1 (DRONE) | v2 (DRONE+Local) | v1 vs Dense | v2 vs Dense |
|------|-------|-----------|-----------------|-------------|-------------|
| **Accuracy** | 92.63% | 88.17% | 80.02% | -4.46% | -12.61% |
| **Latency** | 65.01 ms | 307.8 ms | 296.4 ms | +4.7x slower | +4.6x slower |
| **Peak Mem** | 360.2 MB | 714.9 MB | 1147.1 MB | +1.99x | +3.18x |
| **Model Size** | ~440 MiB | 254.8 MiB | 254.8 MiB | -42.1% | -42.1% |

**结论**:
- **v1** 是目前最佳的压缩方案（准确率和模型大小平衡）
- **v2** 当前不推荐（准确率损失过大）
- **Dense** 仍然是速度和内存最优的选择（如果可接受模型大小）

---

## 为什么 v2 在 encoder 上效果不好？

### 理论分析

1. **Encoder vs Decoder 架构差异**:
   - Decoder LLM: 自回归，每层输出是下一层输入的唯一来源
   - Encoder: 双向注意力，每层输出会被多个下游任务使用
   - Local update 可能破坏了 encoder 的表征能力

2. **Calibration data 不足**:
   - 只用 4 个 batch (128 samples)
   - Encoder 的表征空间可能需要更多样本覆盖
   - Decoder 是逐 token 生成，每个 token 都是一个样本

3. **优化目标不匹配**:
   - v2 local update 最小化逐层的 IO 误差
   - Encoder 的最终目标是整体的序列表征
   - 逐层优化可能导致误差累积

4. **只更新 FFN**:
   - v2 只对 FFN 层做 local update
   - Attention 层仍然使用初始的 DRONE 分解
   - FFN 和 Attention 的不匹配可能导致性能下降

### 实验验证建议

1. **增加 calibration batch**:
   ```python
   max_batches=16  # instead of 4
   ```

2. **对 Attention 也做 local update**:
   ```python
   svdllm_v2_local_update_attn(...)
   ```

3. **多轮迭代**:
   ```python
   for _ in range(3):
       svdllm_v2_local_update_ffn_only(...)
   ```

4. **使用更大的 ridge**:
   ```python
   ridge=1e-4  # instead of 1e-6
   ```

---

## 下一步工作

### 立即可做

1. ✅ **使用 v1** 进行所有后续实验
2. ✅ **放弃 v2** local update（至少在 encoder 上）
3. ✅ **测试不同 ratio** (0.3, 0.5, 0.7) 对 v1 的影响

### 短期探索

4. 🔬 **调试 v2**:
   - 增加 calibration batch 到 16+
   - 对 Attention 层也应用 local update
   - 记录每层的 reconstruction error

5. 🔬 **验证 v1 在其他模型上的效果**:
   - RoBERTa
   - ModernBERT
   - DistilBERT

### 长期研究

6. 📊 **真正实现 v2 公式**:
   - 当前两个版本都用 v1 rank 公式
   - 实现 `R = int(min(m, n) * ratio)` 并对比
   - 可能 v2 公式 + v1 方法（无 local update）效果更好

7. 📊 **设计新的 encoder-friendly local update**:
   - 考虑双向注意力特性
   - 使用 validation loss 作为优化目标
   - 端到端微调而非逐层更新

---

## 参考

- SVD-LLM v1: 标准 DRONE/Whiten 分解
- SVD-LLM v2: DRONE + teacher-student local update
- 代码: `src/encoders/BERTWhiting/profile_svdllm_v1.py` 和 `profile_svdllm_v2.py`
