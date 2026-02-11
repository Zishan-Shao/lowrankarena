# Peak Memory 统计位置分析

## 问题

各个实现中的峰值内存统计位置是否正确？是否混入了训练/准备阶段的内存？

---

## src 实现分析 (profile_svdllm_v1.py / v2.py)

### 代码流程

```python
# Line 462-464 (v1) / 588-590 (v2): 第一次 reset
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()  # ← Reset #1
torch.cuda.synchronize()

# Line 467-468: Calibration (分配临时内存)
covs = calibrate_covariances(model, loader, device, max_batches=4)
# 这里会分配协方差矩阵: 12 layers × 4 matrices × (768² or 3072²)
# 估算: 12 × 4 × (768² × 4 bytes) ≈ 113 MB

# Line 471-482: Create SVDBlocks (分配模型参数)
for i, layer in enumerate(model.bert.encoder.layer):
    blk = SVDBlock(...)  # 分配 Pq, Vq, Pk, Vk, Pv, Vv, U1, V1, U2, V2
    model.layer[i] = LayerShim(blk)
# 模型参数: 254.8 MiB (已报告)

# v2 only: Line 609-618: Local update (分配大量临时内存)
model = svdllm_v2_local_update_ffn_only(...)
# 临时内存: teacher model + IO pairs + accumulators
# 估算: ~440 MB (teacher) + ~200 MB (临时) ≈ 640 MB

# Line 505 (v1) / 643 (v2): 第一次峰值统计
with_act = torch.cuda.max_memory_allocated() / 1024**2
print(f"low-rank model storage with GPU redundancy: {with_act:.1f} MiB")
# ⚠️ 这包含了上面所有步骤的峰值！

# Line 512 (v1) / 650 (v2): 第二次峰值统计
acc, peak_lr, t = acc_peak_time(model, loader, device, task_name)
print(f"... | peak ={peak_lr:6.1f} MiB | ...")
# ✅ 这是推理时的峰值（acc_peak_time 内部会 reset）
```

### acc_peak_time 内部 (Line 360-385)

```python
def acc_peak_time(mdl, loader, device, task_name: str):
    mdl.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()  # ← Reset #2（关键！）

    # ... 运行推理
    for batch in loader:
        logits = mdl(...)
        preds = torch.argmax(logits, -1)
        total += metric.compute(...)

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**2  # ← 只包含推理峰值
    return acc, peak, ms_per_batch
```

### 两次峰值统计的区别

| 统计点 | 位置 | 包含内容 | 值 (v1) | 值 (v2) | 是否报告 |
|-------|------|---------|---------|---------|---------|
| **#1** | Line 505/643 | Calibration + SVDBlock + (v2 local update) | 1212.6 MB | 1630.7 MB | ❌ 仅打印 |
| **#2** | Line 512/650 | 仅推理 | **714.9 MB** | **1147.1 MB** | ✅ 最终报告 |

**结论**:
- ✅ **最终报告的峰值（714.9 / 1147.1 MB）统计位置正确**
- 这些值来自 acc_peak_time，在 reset 后只测量推理
- "low-rank model storage with GPU redundancy" 只是中间诊断信息

---

## eval_encoder 实现分析 (run_encoder_benchmark.py)

### measure_performance 函数 (Line 658-714)

```python
def measure_performance(model, loader, device, warmup_steps, measure_steps,
                        num_runs=1, full_validation=False):
    model.eval()

    if full_validation:
        # Full validation mode
        batch = next(iter(loader))
        for _ in range(warmup_steps):
            model(...)  # Warmup

        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()  # ← Reset

        # Measure over full dataset
        for batch in loader:
            model(...)

        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        return latency_ms, throughput_sps, peak_mem_mb

    else:
        # Standard mode (固定步数)
        for _ in range(warmup_steps):
            model(...)  # Warmup

        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()  # ← Reset

        for _ in range(measure_steps):
            model(...)  # Measure

        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        return latency_ms, throughput_sps, peak_mem_mb
```

### 特点

1. ✅ **Warmup 后才 reset** - 正确
2. ✅ **只测量推理峰值** - 正确
3. ✅ **不包含模型加载/压缩的内存** - 正确

---

## 问题：为什么 v2 比 v1 峰值高 60%？

### 数据对比

| 方法 | 推理峰值 | 模型大小 | 差异 |
|------|----------|----------|------|
| **v1** | 714.9 MB | 254.8 MiB | 基准 |
| **v2** | 1147.1 MB | 254.8 MiB | +60.5% |

**问题**: 模型大小相同（都是 254.8 MiB），为什么推理峰值差这么多？

### 可能原因分析

#### 原因 1: 内存碎片 ❓

**假设**: v2 的准备阶段分配了更多内存（teacher + local update），释放后留下碎片

**验证**:
```python
# 在 acc_peak_time 之前
torch.cuda.empty_cache()  # 已经有了
torch.cuda.synchronize()  # 确保释放完成

# 检查碎片
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
```

如果 Reserved >> Allocated，说明有大量碎片。

#### 原因 2: V 矩阵的数值特性 ❓

**假设**: v2 local update 改变了 V 的数值，导致推理时的中间 tensor 更大

**分析**: 不太可能，因为 tensor 大小由模型结构决定，不受权重数值影响。

#### 原因 3: 测量时机问题 ⚠️

**发现**: acc_peak_time 在 v2 中可能**没有完全释放 teacher 模型**！

```python
# v2 的流程
teacher = BertForSequenceClassification.from_pretrained(...)  # Line 574
model = svdllm_v2_local_update_ffn_only(student=model, teacher=teacher, ...)  # Line 610
# ... teacher 还在内存中！

acc, peak_lr, t = acc_peak_time(model, loader, device, task_name)  # Line 650
# ⚠️ teacher 是否还占用内存？
```

**检查**: teacher 是局部变量，在 main 函数中定义，acc_peak_time 调用时仍在作用域内！

```python
def main():
    # Line 574: teacher 定义
    teacher = BertForSequenceClassification.from_pretrained(...)

    # Line 610-618: local update
    model = svdllm_v2_local_update_ffn_only(student=model, teacher=teacher, ...)

    # Line 650: evaluate
    acc, peak_lr, t = acc_peak_time(model, loader, ...)
    # ⚠️ teacher 还在作用域中，可能还在 GPU 上！
```

**验证**: 检查 teacher 是否在 GPU 上

```python
# 在 Line 619 之后（local update 完成后）
print(f"Teacher device: {teacher.device}")
print(f"Teacher memory: {sum(p.numel() * p.element_size() for p in teacher.parameters()) / 1024**2:.1f} MB")

# 释放 teacher
del teacher
torch.cuda.empty_cache()
```

#### 原因 4: 实际的峰值差异（正常） ✅

**假设**: v2 的 V 矩阵确实导致推理时需要更多激活内存

**可能机制**:
- v2 local update 后的 V 可能产生更大的激活值（范数更大）
- 导致在 attention/FFN 计算时需要更多临时空间
- 但这不太可能导致 60% 的差异

---

## 验证实验

### 实验 1: 检查 teacher 是否释放

在 v2 的 Line 619 添加：

```python
# After local update
print(f"\n[DEBUG] Before del teacher:")
print(f"  Teacher on GPU: {next(teacher.parameters()).is_cuda}")
print(f"  GPU allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

del teacher
torch.cuda.empty_cache()

print(f"[DEBUG] After del teacher:")
print(f"  GPU allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
```

**预期**: 如果 teacher 在 GPU 上，应该看到 ~440 MB 的释放。

### 实验 2: 对比 v1 和 v2 的激活内存

```python
# 在 acc_peak_time 的 for batch in loader 循环中
for batch in loader:
    before_alloc = torch.cuda.memory_allocated()
    logits = mdl(...)
    after_alloc = torch.cuda.memory_allocated()

    activation_mem = (after_alloc - before_alloc) / 1024**2
    print(f"Activation memory: {activation_mem:.1f} MB")
```

**预期**: v2 的激活内存应该和 v1 相近（模型结构相同）。

### 实验 3: 检查内存碎片

```python
# 在 acc_peak_time 开始时
torch.cuda.empty_cache()
torch.cuda.synchronize()

allocated = torch.cuda.memory_allocated() / 1024**2
reserved = torch.cuda.memory_reserved() / 1024**2
print(f"Before inference: Allocated={allocated:.1f} MB, Reserved={reserved:.1f} MB")
print(f"Fragmentation: {reserved - allocated:.1f} MB")
```

**预期**: 如果碎片 >200 MB，说明内存碎片是主要原因。

---

## 修复建议

### 修复 1: 显式释放 teacher (推荐)

在 v2 的 Line 619 添加：

```python
# After local update, explicitly delete teacher
del teacher
torch.cuda.empty_cache()
torch.cuda.synchronize()

print(f"[INFO] Teacher model released from GPU")
```

### 修复 2: 在 acc_peak_time 前强制清理

在 Line 649 添加：

```python
# Before evaluation, ensure clean GPU state
import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

print(f"GPU memory before eval: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
```

### 修复 3: 使用 CPU teacher (如果内存是瓶颈)

在 Line 574 修改：

```python
# Keep teacher on CPU to save GPU memory
teacher = BertForSequenceClassification.from_pretrained(MODEL_DIR, config=cfg)
teacher = teacher.to('cpu').eval()  # ← Keep on CPU
```

然后在 local update 的 hook 中：

```python
def hook_teacher_output(mod, inp, out):
    Y_teacher = out.to(device)  # Copy to GPU only when needed
```

**代价**: local update 会变慢（需要 CPU-GPU 传输）

---

## 结论

### Peak Memory 统计位置

| 实现 | Reset 位置 | 统计位置 | 是否正确 | 包含内容 |
|------|-----------|---------|---------|---------|
| **src v1/v2** | acc_peak_time 内部 | acc_peak_time 返回值 | ✅ 正确 | 仅推理 |
| **eval_encoder** | warmup 后 | measure_performance 返回值 | ✅ 正确 | 仅推理 |

### v2 峰值异常高的原因

**最可能**: teacher 模型未释放 (440 MB)

**验证方法**:
1. 在 local update 后检查 teacher 是否在 GPU
2. 显式 `del teacher` 并测量释放的内存
3. 重新运行 acc_peak_time 看峰值是否降低

**预期修复后的峰值**: ~700-750 MB (接近 v1 的 714.9 MB)

---

## 下一步

1. ✅ **确认统计位置正确** - 已确认，所有实现都在 reset 后统计
2. 🔬 **调查 v2 峰值异常** - 需要实验验证 teacher 释放问题
3. 🔧 **修复 v2** - 显式释放 teacher 并重新测试
4. 📊 **更新对比文档** - 修复后重新对比 v1 vs v2
