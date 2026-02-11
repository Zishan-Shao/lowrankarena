# v2 Local Update 数据流问题分析

## 问题发现

v2 的 local update 实现存在**数据统计位置不匹配**的问题！

---

## 当前实现的数据流

### 代码位置: `profile_svdllm_v2.py:457-488`

```python
# Step 1: 固定 student 的 U
U1 = blk.U1.detach()  # student 的 U1 [dm, r]
U2 = blk.U2.detach()  # student 的 U2 [dff, r]

# Step 2: Hook 注册在 teacher 上
def hook_ffn1(mod, inp, out):
    X = inp[0]      # ⚠️ teacher 的输入
    Y = out         # ⚠️ teacher 的输出
    Z = X @ U1      # ⚠️ X_teacher @ U_student
    A1 += Z.t() @ Z
    B1 += Z.t() @ (Y - b1)

h1 = t_layer.intermediate.dense.register_forward_hook(hook_ffn1)
h2 = t_layer.output.dense.register_forward_hook(hook_ffn2)

# Step 3: 只运行 teacher
_ = teacher(input_ids=..., attention_mask=...)

# Step 4: 求解 V
V_new = solve(A + ridge*I, B)
blk.V1.data.copy_(V_new)
```

### 数据来源分析

| 变量 | 来源 | 说明 |
|------|------|------|
| **X** | `inp[0]` from teacher hook | ❌ teacher.intermediate.dense 的**输入** |
| **Y** | `out` from teacher hook | ✅ teacher.intermediate.dense 的**输出** |
| **U1** | `blk.U1` from student | ✅ student 的 U 矩阵 |
| **Z** | `X @ U1` | ❌ **X_teacher @ U_student** |

---

## 问题分析

### 问题 1: X 来源不匹配

**训练时**（local update）:
```python
Z = X_teacher @ U_student
V_new = solve(Z^T Z, Z^T Y_teacher)
```

**推理时**（实际使用）:
```python
Z = X_student @ U_student  # ⚠️ X 来源不同！
Y_student = Z @ V + b
```

**为什么 X_teacher ≠ X_student?**

```
Input → [Layer 0] → [Layer 1] → ... → [Layer i-1] → [Layer i: FFN]
                                                          ↑
                                                          X

Teacher path:
Input → teacher.layer[0] → ... → teacher.layer[i-1] → X_teacher

Student path:
Input → student.layer[0] → ... → student.layer[i-1] → X_student
        (SVD blocks)              (SVD blocks)
```

- Teacher 的前面层是 **dense**
- Student 的前面层是 **SVD blocks**（已压缩）
- 即使输入相同，经过不同的层后，**X_teacher ≠ X_student**！

### 问题 2: 优化目标错位

**当前优化的是**:
```
min_V ||Y_teacher - (X_teacher @ U_student) @ V||²
```

**实际需要的是**:
```
min_V ||Y_teacher - (X_student @ U_student) @ V||²
```

或者更准确地说，应该是:
```
min_V ||Y_student_target - (X_student @ U_student) @ V||²
```

其中 `Y_student_target` 是我们希望 student 在这一层输出的目标（可以是 teacher 的输出）。

---

## 为什么仍然有 87.72% 的准确率？

虽然存在这个问题，但 v2 仍然达到了 87.72% 的准确率（虽然比 v1 的 88.17% 低）。

**可能的原因**:

1. **X 的分布相似性**:
   - 虽然 X_teacher ≠ X_student，但它们的**统计分布可能相似**
   - 都是经过 LayerNorm 后的表征，范数和方向分布可能接近
   - V 的优化仍然能部分泛化到 X_student

2. **U 的主导作用**:
   - DRONE 优化的 U 已经保留了输入的关键信息
   - V 的主要作用是输出重建，对 X 的依赖可能较弱
   - 即使 X 有偏差，只要 U 正确，V 的调整仍有帮助

3. **Ridge 正则化**:
   - ridge=1e-4 提供了正则化，避免过拟合到特定的 X_teacher
   - 使得学到的 V 更通用

4. **误差累积的平衡**:
   - 虽然每层的 X 不匹配，但误差可能在多层间相互抵消
   - 最终的分类头可能对中间层的小偏差有一定容忍度

---

## 正确的实现方式

### 方案 A: Layer-wise Feature Alignment

**思路**: 先对齐 student 和 teacher 的中间特征，再做 local update

```python
# Step 1: 同时运行 student 和 teacher
for batch in loader:
    s_out = student(input_ids, attention_mask, output_hidden_states=True)
    t_out = teacher(input_ids, attention_mask, output_hidden_states=True)

    for i, (s_layer, t_layer) in enumerate(zip(student.layers, teacher.layers)):
        # X_student: student 在第 i 层的输入
        X_student = s_out.hidden_states[i]

        # Y_teacher: teacher 在第 i 层 FFN 的输出
        # 需要手动 forward FFN 部分
        Y_teacher = t_layer.intermediate.dense(X_student)  # ⚠️ 还是用 X_student

        # 或者使用 teacher 在第 i 层的输出
        Y_teacher_full = t_out.hidden_states[i+1]
```

**问题**: 这样的话还是需要用 X_student，那 teacher 的意义是什么？

### 方案 B: End-to-End Distillation

**思路**: 不做 layer-wise update，而是端到端优化

```python
# 不固定 V，而是端到端微调
student.requires_grad_(True)
optimizer = AdamW(student.parameters(), lr=1e-5)

for batch in loader:
    s_logits = student(input_ids, attention_mask).logits
    t_logits = teacher(input_ids, attention_mask).logits

    # KL divergence loss
    loss = F.kl_div(
        F.log_softmax(s_logits / T, dim=-1),
        F.softmax(t_logits / T, dim=-1),
        reduction='batchmean'
    ) * (T ** 2)

    loss.backward()
    optimizer.step()
```

**优势**: 直接优化最终目标，不受中间层不匹配的影响

### 方案 C: Self-Distillation (当前可能最接近的正确做法)

**思路**: 不使用 teacher，而是在 student 自己的 forward 中收集 IO pairs

```python
# 在 student 上注册 hook，收集未压缩的 W 和压缩后的 U, V 的差异
def hook_ffn1(mod, inp, out):
    X_student = inp[0]  # ✅ student 的输入

    # 计算理想的输出（用原始 W，如果有的话）
    # 或者用 DRONE 初始化的输出作为 target
    Y_drone = X_student @ U_init @ V_init + b

    # 用当前的 U 计算 Z
    Z = X_student @ U_current

    # 优化 V 使得 Z @ V 接近 Y_drone
    ...
```

**问题**: 这样就没有用到 teacher 了，本质上是在优化 DRONE 的重建误差。

---

## 实验验证

### 假设验证: X_teacher 和 X_student 的差异

让我们测量一下实际的差异：

```python
# 同时运行 student 和 teacher
for batch in loader:
    with torch.no_grad():
        s_hidden = []
        t_hidden = []

        def hook_s(mod, inp, out):
            s_hidden.append(inp[0].detach())
        def hook_t(mod, inp, out):
            t_hidden.append(inp[0].detach())

        h_s = student.layer[i].intermediate.dense.register_forward_pre_hook(hook_s)
        h_t = teacher.layer[i].intermediate.dense.register_forward_pre_hook(hook_t)

        _ = student(input_ids, attention_mask)
        _ = teacher(input_ids, attention_mask)

        h_s.remove()
        h_t.remove()

        # 计算差异
        X_s = s_hidden[0]
        X_t = t_hidden[0]

        diff_norm = torch.norm(X_s - X_t) / torch.norm(X_t)
        cos_sim = F.cosine_similarity(X_s, X_t, dim=-1).mean()

        print(f"Layer {i}: diff_norm={diff_norm:.4f}, cos_sim={cos_sim:.4f}")
```

**预期结果**:
- 如果 diff_norm 很大（>0.5），说明 X 差异显著，v2 的做法有问题
- 如果 cos_sim 很小（<0.8），说明方向差异大
- 如果两者都接近（diff_norm < 0.1, cos_sim > 0.95），说明 X 的统计相似，v2 的做法可能侥幸有效

---

## 为什么 v1 没有这个问题？

v1 (DRONE only) 不做 local update，所以：

1. **初始分解时**:
   - 使用 calibration data 在 **student 自己**的 forward 中收集协方差
   - X 就是 student 的实际输入
   - ✅ 训练和推理一致

2. **无需 teacher**:
   - DRONE 直接优化 ||X^T (W - UV)||_F
   - W 是原始权重，X 是 student 的输入
   - ✅ 完全自洽

---

## 修复建议

### 短期（快速修复）

**修改 v2 的 hook 位置**:

```python
# 同时在 student 和 teacher 上 hook
def collect_io_pairs():
    X_student_list = []
    Y_teacher_list = []

    def hook_student_input(mod, inp):
        X_student_list.append(inp[0].detach())

    def hook_teacher_output(mod, inp, out):
        Y_teacher_list.append(out.detach())

    h_s = student_layer.intermediate.dense.register_forward_pre_hook(hook_student_input)
    h_t = teacher_layer.intermediate.dense.register_forward_hook(hook_teacher_output)

    for batch in loader:
        # 同时运行
        _ = student(input_ids, attention_mask)
        _ = teacher(input_ids, attention_mask)

    h_s.remove()
    h_t.remove()

    # 现在 X 来自 student, Y 来自 teacher
    for X_s, Y_t in zip(X_student_list, Y_teacher_list):
        Z = X_s @ U_student
        A += Z.t() @ Z
        B += Z.t() @ Y_t
```

**问题**: 需要同时运行两个模型，内存翻倍（可能 >2GB）

### 中期（推荐）

**放弃 layer-wise local update，改用端到端 distillation**:

```python
# 简单的 KL divergence loss
for batch in loader:
    s_logits = student(...).logits
    t_logits = teacher(...).logits
    loss = kl_div(s_logits, t_logits)
    loss.backward()
    optimizer.step()
```

**优势**:
- 优化最终目标（分类准确率）
- 不受中间层不匹配影响
- 实现简单

### 长期（研究方向）

**设计适合 encoder 的 local update 方法**:

1. **Feature alignment**: 先对齐中间表征，再做 distillation
2. **Self-distillation**: 不用 teacher，优化 SVD 重建误差
3. **Multi-stage**: 分阶段优化（先 U，再 V，最后端到端微调）

---

## 结论

1. **v2 当前实现确实存在数据流问题**:
   - X 来自 teacher，但推理时用的是 student 的 X
   - 训练-推理不一致

2. **为什么仍有 87.72% 准确率**:
   - X 的统计分布可能相似
   - U 的质量（DRONE）更关键
   - Ridge 正则化提供泛化

3. **v1 (88.17%) 仍然更好**:
   - 没有这个问题
   - 训练-推理完全一致
   - 实现简单

4. **修复建议**:
   - 短期：修改 hook 同时收集 X_student 和 Y_teacher（内存成本高）
   - 中期：改用端到端 distillation（推荐）
   - 长期：设计新的 encoder-friendly local update

**最终建议**: 继续使用 v1，放弃 v2 的 local update（至少在当前实现下）
