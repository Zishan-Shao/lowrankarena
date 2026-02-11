# SVD-LLM 标准基线对比（BERT SST-2）

## 测试配置

**模型**: `textattack/bert-base-uncased-sst-2`
**任务**: SST-2 Sentiment Classification
**数据集**: GLUE SST-2 validation (872 samples, 28 batches)
**Batch size**: 32
**Sequence length**: 256
**Ratio**: 0.5
**Device**: CUDA
**测试日期**: 2026-02-10

## Rank 配置（相同）

```python
def rank_from_ratio(m, n, ratio):
    return int(m * n * ratio / (m + n))

RANK_ATTN = rank_from_ratio(768, 64, 0.5)   = 29   # per-head Q/K/V
RANK_FF   = rank_from_ratio(768, 3072, 0.5) = 307  # FFN intermediate
RANK_WO   = rank_from_ratio(768, 768, 0.5)  = 192  # Attention output
```

## 完整结果对比

| 方法 | Accuracy | Peak Mem | Latency | Model Size | 说明 |
|------|----------|----------|---------|------------|------|
| **Dense Baseline** | **92.63%** | 360.2 MB | 65.0 ms | ~440 MiB | 未压缩 |
| **SVD-LLM v1** | **88.17%** | 714.9 MB | 322.4 ms | 254.8 MiB | DRONE only |
| **SVD-LLM v2** | **87.61%** | 728.3 MB | 310.3 ms | 254.8 MiB | DRONE + local update |

## 方法描述

### Dense Baseline
- 标准 BERT-base 模型，未压缩
- 最高准确率但模型大
- 最快推理速度

### SVD-LLM v1: DRONE Only

**流程**:
1. Calibrate input covariances (4 batches)
   - Per-layer covariances: C_qkvm, C_wo, C_ff1, C_ff2
2. DRONE factorization for each Linear layer:
   ```
   S = chol(C)           # Cholesky decomposition
   A = W^T @ S           # Whitened weight
   U, Σ, V = SVD(A)      # Truncate to rank k
   U_data = S^{-T} @ V @ Σ^{1/2}   # [d_in, k]
   V_data = Σ^{1/2} @ U^T          # [k, d_out]
   ```
3. Replace layers with SVD blocks
4. Done - no further updates

**特点**:
- ✅ 一次性分解，简单高效
- ✅ 准确率最高（88.17%）
- ✅ 内存占用相对较低（714.9 MB）
- ✅ 实现稳定

### SVD-LLM v2: DRONE + Local Update

**流程**:
1. Apply DRONE factorization（同 v1）
2. **Extra step**: Local update（FFN only）
   - Load dense teacher model
   - For each layer:
     - Fix U1, U2 from student (DRONE初始化)
     - Collect teacher IO pairs: (X, Y)
       - X = teacher layer input
       - Y = teacher layer output
     - Solve for V1, V2:
       ```
       Z = X @ U               # [N, r]
       A = Z^T @ Z             # [r, r]
       B = Z^T @ (Y - b)       # [r, d_out]
       V_new = solve(A + λI, B)
       ```
     - Update student V matrices
   - Free teacher model

**特点**:
- ⚠️ 准确率略低于 v1（-0.56%）
- ⚠️ 内存占用稍高（+13.4 MB）
- ✅ 速度略快（+3.8%）
- ⚠️ 需要额外的 teacher 模型和 local update 步骤

## 性能分析

### 准确率对比

| 方法 | Accuracy | vs Dense | vs v1 | 说明 |
|------|----------|----------|-------|------|
| Dense | 92.63% | - | +4.46% | 基准 |
| **v1** | **88.17%** | -4.46% | **0.00%** | 最佳压缩 |
| v2 | 87.61% | -5.02% | -0.56% | 略逊 v1 |

**关键发现**:
- v2 local update **并未提升准确率**，反而下降 0.56%
- 可能原因：
  1. Teacher-student 数据流不完全对齐
  2. Local update 只用 4 batches，样本可能不足
  3. V 更新虽然比 U 更新好，但仍不如 v1 的完整 DRONE

### 速度对比

| 方法 | Latency | vs Dense | vs v1 | Throughput |
|------|---------|----------|-------|------------|
| Dense | 65.0 ms | - | -79.8% | ~492 sps |
| v1 | 322.4 ms | +4.96x | baseline | ~99 sps |
| **v2** | **310.3 ms** | +4.77x | **-3.8%** | ~103 sps |

**关键发现**:
- v2 速度略快于 v1（3.8%提升）
- 但相比 dense 都慢 4.7-5.0x
- 速度提升不足以弥补准确率损失

### 内存对比

| 方法 | Peak Mem | vs Dense | vs v1 | Model Size |
|------|----------|----------|-------|------------|
| Dense | 360.2 MB | - | -49.6% | ~440 MiB |
| **v1** | **714.9 MB** | +1.99x | **baseline** | 254.8 MiB |
| v2 | 728.3 MB | +2.02x | +1.9% | 254.8 MiB |

**关键发现**:
- v2 峰值内存略高于 v1（13.4 MB差异）
- 模型大小相同（都是 254.8 MiB）
- Dense 反而内存最低（因为没有中间 SVD 计算）

## 为什么 v2 没有提升？

### 理论预期
SVD-LLM 论文中 v2 在 decoder LLM 上有效，通过 teacher-student distillation 微调 V 矩阵。

### Encoder 上的问题

1. **数据流不对齐**
   ```python
   # Training (local update)
   X = teacher.layer[i].input    # Teacher 的输入
   Y = teacher.layer[i].output   # Teacher 的输出
   V = solve(X @ U_student, Y)   # X 来自 teacher

   # Inference
   X = student.layer[i-1].output  # Student 的输入
   Y = X @ U @ V                  # X 来自 student
   ```
   - Training 和 inference 的 X 分布不同
   - 虽然都是同样的样本，但经过不同的前置层

2. **样本数量限制**
   - 只用 4 batches (128 samples)
   - Encoder 的双向注意力可能需要更多样本覆盖

3. **V 更新的局限**
   - 虽然固定 U 更新 V 比固定 V 更新 U 好
   - 但仍然不如 v1 的完整 DRONE 优化
   - DRONE 同时考虑了 U 和 V 的联合优化

4. **Encoder vs Decoder 差异**
   - Decoder LLM: 自回归，逐层依赖性强
   - Encoder: 双向注意力，层间独立性强
   - Local update 可能更适合 decoder 架构

## 使用建议

### 推荐顺序

1. **首选: SVD-LLM v1 (DRONE only)** ⭐⭐⭐⭐⭐
   - 准确率最高（88.17%）
   - 实现简单，稳定
   - 内存占用合理
   - **适用场景**: 所有 BERT encoder 模型压缩

2. **Dense baseline** ⭐⭐⭐⭐ (如果资源允许)
   - 准确率最高（92.63%）
   - 速度最快
   - 但模型大（440 MiB）
   - **适用场景**: 资源充足，追求最高准确率

3. **SVD-LLM v2** ⭐⭐⭐ (不推荐用于 encoder)
   - 准确率低于 v1
   - 实现复杂
   - 需要额外 teacher 模型
   - **适用场景**: 仅用于研究对比

### 不同场景选择

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| **生产部署** | v1 | 准确率最高，稳定 |
| **资源充足** | Dense | 最佳性能 |
| **研究对比** | v1 + v2 | 完整基线 |
| **极限压缩** | v1 with lower ratio | 可调节 |

## 改进方向（如果必须用 v2）

基于诊断分析，如果一定要改进 v2：

1. **增加 calibration samples**
   ```python
   max_batches = 16  # instead of 4
   ```
   - 测试结果：87.28% → 87.39%（+0.11%）
   - 仍然不如 v1

2. **修复 embeddings 对齐**
   ```python
   H0 = student.bert.embeddings(...)  # instead of teacher
   ```
   - 配合 max_batches=16 效果更好
   - 但单独使用反而变差

3. **联合优化 U 和 V**
   - 目前只更新 V
   - 可以尝试交替更新
   - 但会增加复杂度

4. **端到端微调**
   - 在 local update 后进行 few-shot fine-tuning
   - 可能修正误差

## 结论

**核心发现**:
1. **v1 (DRONE) 是最佳的 encoder 压缩方法**
2. **v2 local update 在 encoder 上无效甚至有害**
3. **Dense baseline 仍然是准确率和速度的最佳选择**（如果可接受模型大小）

**原因分析**:
- v2 设计针对 decoder LLM
- Encoder 的双向注意力和层间独立性与 decoder 不同
- Local update 的数据流不对齐问题在 encoder 上更严重

**最终建议**:
- **研究基线**: 使用 v1
- **生产部署**: 使用 v1 或 dense（取决于资源）
- **不推荐**: v2 用于 encoder 模型

---

**文档版本**: 1.0
**最后更新**: 2026-02-10
**测试平台**: CUDA GPU
**测试模型**: BERT-base SST-2
