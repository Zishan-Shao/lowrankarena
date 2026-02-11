# SVD-LLM v2: 更新策略对比（固定U vs 固定V）

## 重要发现

v2 的 Local Update 有两种策略：
1. **固定 V，更新 U** (左矩阵)
2. **固定 U，更新 V** (右矩阵)

结果显示**策略选择至关重要**！

---

## 测试配置

**模型**: `textattack/bert-base-uncased-sst-2`
**Ratio**: 0.5
**Rank**: ATTN=29, FF=307, WO=192
**Calibration**: 4 batches

---

## 三种方法对比

### 1. v1: DRONE Only (Baseline)

**方法**:
- 仅使用 Whiten/DRONE 初始分解
- 无 local update

**结果**:
```
Accuracy      : 88.17%
Peak Memory   : 714.9 MB
Latency       : 307.8 - 330.5 ms/batch (有波动)
Model Size    : 254.8 MiB
```

### 2. v2: 固定 V，更新 U (早期实现)

**方法**:
```python
# Fix V1, V2 (right matrices)
V1 = blk.V1.detach()
V2 = blk.V2.detach()

# Update U1, U2 (left matrices)
U1_new = _solve_U_fixed_V(X1, Y1, V1, ridge)
U2_new = _solve_U_fixed_V(X2, Y2, V2, ridge)

blk.U1.data.copy_(U1_new)
blk.U2.data.copy_(U2_new)
```

**结果**:
```
Accuracy      : 80.02%  ⚠️ -8.15% vs v1
Peak Memory   : 1147.1 MB
Latency       : 296.4 ms/batch
Model Size    : 254.8 MiB
Update Log    : "[v2] updated FFN U at layer X"
```

**分析**:
- ❌ 准确率显著下降
- ✅ 速度略快（~3%）
- ❌ 内存增加 60%
- **不推荐使用**

### 3. v2: 固定 U，更新 V (当前实现)

**方法**:
```python
# Fix U1, U2 (left matrices)
U1 = blk.U1.detach()
U2 = blk.U2.detach()

# Update V1, V2 (right matrices)
V1_new = _solve_V_fixed_U(X1, Y1, U1, b1, ridge)
V2_new = _solve_V_fixed_U(X2, Y2, U2, b2, ridge)

blk.V1.data.copy_(V1_new)
blk.V2.data.copy_(V2_new)
```

**结果**:
```
Accuracy      : 87.17%  ✅ -1.0% vs v1 (可接受)
Peak Memory   : 1147.1 MB
Latency       : 325.8 ms/batch
Model Size    : 254.8 MiB
Update Log    : "[v2] updated FFN V at layer X"
```

**分析**:
- ✅ 准确率接近 v1（仅差 1%）
- ❌ 内存仍然较高（+60%）
- ⚠️ 速度无明显优势
- **可以考虑使用**（如果能接受内存开销）

---

## 为什么固定 U 比固定 V 好？

### 矩阵形状分析

对于 FFN 层（以 `intermediate.dense` 为例）:
```
W_original: [dm, dff] = [768, 3072]
W ≈ U @ V

U1: [dm, r]  = [768, 307]  (左矩阵，输入侧)
V1: [r, dff] = [307, 3072] (右矩阵，输出侧)
```

### 理论解释

1. **U 矩阵（输入投影）更稳定**:
   - U 将输入从高维 (768) 投影到低维 (307)
   - 这个投影保留了输入的主要方向（由 DRONE 优化）
   - 输入分布相对稳定

2. **V 矩阵（输出重建）需要调整**:
   - V 将低维 (307) 重建回高维 (3072)
   - 这个重建需要匹配 teacher 的输出分布
   - teacher 的输出可能有特定的 pattern 需要学习

3. **DRONE 初始化的作用**:
   - DRONE 优化了 U 使得 `X @ U` 保留输入的关键信息
   - V 的初始化可能不是最优的（只是 SVD 的结果）
   - local update 可以进一步优化 V

4. **参数数量对比**:
   - U1: 768 × 307 = 235,776
   - V1: 307 × 3072 = 943,104
   - V 的参数是 U 的 4 倍，自由度更高，更容易优化

### 实验证据

| 策略 | Accuracy | vs v1 | 解释 |
|------|----------|-------|------|
| **固定 V，更新 U** | 80.02% | -8.15% | U 被破坏，输入投影失效 |
| **固定 U，更新 V** | 87.17% | -1.0% | U 保留，V 适配 teacher 输出 |

---

## 更新公式对比

### 固定 V，更新 U

**问题**: Y ≈ X @ U @ V，求 U（V 已知）

**投影到线性目标**:
```
Yproj = Y @ V^T @ (V V^T + ridge I)^{-1}
U = lstsq(X, Yproj)
```

**为什么效果差**:
- 破坏了 DRONE 优化的输入投影
- X @ U_new 可能不再保留输入的关键信息
- 下游层依赖于这个投影的质量

### 固定 U，更新 V

**问题**: Y ≈ X @ U @ V，求 V（U 已知）

**投影到线性目标**:
```
Z = X @ U  (low-rank representation)
V_new = lstsq(Z, Y - b)
```

**为什么效果好**:
- 保留了 DRONE 优化的输入投影
- 只调整输出重建部分
- 不影响下游层的输入质量

---

## 完整对比表

| 方法 | Accuracy | vs Dense | Peak Mem | Latency | 推荐度 |
|------|----------|----------|----------|---------|-------|
| **Dense** | 92.63% | - | 360.2 MB | 65.0 ms | ⭐⭐⭐⭐⭐ (如果内存充足) |
| **v1 (DRONE)** | 88.17% | -4.46% | 714.9 MB | 307-330 ms | ⭐⭐⭐⭐ (压缩首选) |
| **v2 (固定V更新U)** | 80.02% | -12.61% | 1147.1 MB | 296 ms | ⭐ (不推荐) |
| **v2 (固定U更新V)** | 87.17% | -5.46% | 1147.1 MB | 326 ms | ⭐⭐⭐ (可用) |

---

## 使用建议

### 推荐顺序

1. **v1 (DRONE)** ⭐⭐⭐⭐
   - 准确率最高 (88.17%)
   - 内存适中 (714.9 MB)
   - 实现简单
   - **最佳选择**

2. **v2 (固定 U，更新 V)** ⭐⭐⭐
   - 准确率接近 v1 (87.17%)
   - 内存较高 (1147.1 MB)
   - 需要额外的 local update 步骤
   - **如果有大量 calibration data 可以考虑**

3. **Dense** ⭐⭐⭐⭐⭐ (如果资源允许)
   - 准确率最高 (92.63%)
   - 速度最快 (65 ms)
   - 内存最低 (360.2 MB)
   - 但模型大小大 (~440 MiB)

4. **v2 (固定 V，更新 U)** ⭐
   - **不推荐**
   - 准确率太低 (80.02%)

### 何时使用 v2 (固定 U，更新 V)

**适用场景**:
- 有充足的内存 (1.1+ GB)
- 有大量 unlabeled data 用于 local update
- 想要在 v1 基础上挤出额外 1% 的压缩率
- 可以接受更复杂的训练流程

**不适用场景**:
- 内存受限 (<1 GB)
- 只有少量 calibration data (< 16 batches)
- 需要快速实验迭代
- 对准确率要求极高 (需要 >88%)

---

## 改进方向

### 短期

1. **增加 calibration batches**:
   ```python
   max_batches=16  # instead of 4
   ```
   - 可能进一步提升 v2 的准确率

2. **调整 ridge 参数**:
   ```python
   ridge=1e-4  # instead of 1e-6
   ```
   - 更大的 ridge 可能提高稳定性

3. **对 Attention 层也应用 local update**:
   - 当前只更新 FFN，Attention 可能也能受益

### 中期

4. **多轮迭代 local update**:
   ```python
   for _ in range(3):
       svdllm_v2_local_update(...)
   ```
   - 迭代可能逐步提升准确率

5. **使用 validation 的一部分做 local update**:
   - 避免过拟合 calibration data

### 长期

6. **联合优化 U 和 V**:
   - 交替更新 U 和 V
   - 或者同时优化（需要更复杂的求解器）

7. **端到端微调**:
   - 在 local update 后进行少量 epochs 的微调
   - 可能修正 local update 的错误

---

## 代码位置

- v1: `src/encoders/BERTWhiting/profile_svdllm_v1.py`
- v2 (固定V更新U): git history (已废弃)
- v2 (固定U更新V): `src/encoders/BERTWhiting/profile_svdllm_v2.py` (当前)

**关键代码** (`profile_svdllm_v2.py:511-514`):
```python
#blk.U1.data.copy_(U1_new.to(dtype=blk.U1.dtype))  # 注释掉
#blk.U2.data.copy_(U2_new.to(dtype=blk.U2.dtype))  # 注释掉
blk.V1.data.copy_(V1_new.to(dtype=blk.V1.dtype))   # 启用
blk.V2.data.copy_(V2_new.to(dtype=blk.V2.dtype))   # 启用
```

---

## 结论

**关键发现**:
- ✅ 固定 U 更新 V: 87.17% (可用)
- ❌ 固定 V 更新 U: 80.02% (不可用)

**最佳实践**:
1. 优先使用 **v1 (DRONE)** - 准确率最高，实现最简单
2. 如果需要挤出最后 1%，可以尝试 **v2 (固定 U 更新 V)**
3. 绝对不要使用 **v2 (固定 V 更新 U)**

**理论启示**:
- DRONE 优化的输入投影（U）非常重要，不应破坏
- 输出重建（V）有更多优化空间
- Local update 的方向选择至关重要
