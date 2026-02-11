# 性能检测对比: eval_encoder vs src

## 概述

本文档对比分析 `eval_encoder/run_encoder_benchmark.py` 和 `src/encoders/BERTWhiting/profile_svdllm.py` 中的性能检测实现，识别差异并提供统一建议。

---

## 1. 函数签名对比

### eval_encoder (run_encoder_benchmark.py:654)
```python
@torch.no_grad()
def measure_performance(model, loader, device, warmup_steps, measure_steps, num_runs=1):
    """
    Measure model performance with multiple runs.

    Args:
        num_runs: Number of times to run the measurement. If >1, returns median.

    Returns:
        latency_ms, throughput_sps, peak_mem_mb (median if num_runs > 1)
    """
```

### src (profile_svdllm.py:360)
```python
@torch.no_grad()
def acc_peak_time(mdl, loader, device, task_name: str):
    """
    Benchmark helper that measures accuracy, peak memory, and time in one pass.

    Returns:
        accuracy, peak_mem_mb, ms_per_batch
    """
```

**关键差异**:
- **eval_encoder**: 分离的性能测试（不计算accuracy），支持多次运行取中位数
- **src**: 合并的测试（accuracy + perf），单次运行全validation集

---

## 2. 详细实现对比

### 2.1 Warmup 阶段

| 实现 | Warmup | 说明 |
|------|--------|------|
| **eval_encoder** | ✅ 专门的warmup阶段 (默认10步) | `for _ in range(warmup_steps): model(...)` |
| **src** | ❌ 无warmup | 直接开始计时 |

**影响**:
- eval_encoder 更准确（避免首次CUDA kernel编译开销）
- src 可能受kernel编译和cache预热影响

### 2.2 测量范围

| 实现 | 测量范围 | 批次数 |
|------|----------|--------|
| **eval_encoder** | 固定步数 | `measure_steps` (默认50) |
| **src** | 全数据集 | 整个validation split |

**影响**:
- eval_encoder: 快速、可重复、测量步数可控
- src: 测量完整推理过程，包含所有数据

### 2.3 内存测量

| 实现 | 峰值内存 | 时机 |
|------|----------|------|
| **eval_encoder** | ✅ `torch.cuda.reset_peak_memory_stats()` before measurement | 准确测量推理峰值 |
| **src** | ✅ `torch.cuda.reset_peak_memory_stats()` before measurement | 准确测量推理峰值 |

**相同点**: 两者都正确重置峰值内存统计

### 2.4 时间测量

| 实现 | 时间计算 | 公式 |
|------|----------|------|
| **eval_encoder** | `elapsed * 1000.0 / measure_steps` | 每批次平均延迟 (ms) |
| **src** | `(end - start) * 1000.0 / max(steps, 1)` | 每批次平均延迟 (ms) |

**相同点**: 都计算ms/batch，计算方式相同

### 2.5 吞吐量

| 实现 | 吞吐量 | 单位 |
|------|--------|------|
| **eval_encoder** | ✅ `bs * measure_steps / elapsed` | samples/second |
| **src** | ❌ 不计算 | N/A |

**差异**: eval_encoder 额外提供吞吐量指标

### 2.6 多次运行取中位数

| 实现 | 支持多次运行 | 统计方法 |
|------|------------|----------|
| **eval_encoder** | ✅ `num_runs` 参数 | `statistics.median()` |
| **src** | ❌ 单次运行 | N/A |

**影响**:
- eval_encoder: 更可靠（减少噪声）
- src: 单次运行可能受系统负载影响

---

## 3. Accuracy 评估对比

### 3.1 eval_encoder (evaluate_task:625)

```python
@torch.no_grad()
def evaluate_task(model, loader, task, device):
    """Return (metric_name, metric_value)."""
    metric = load_metric("accuracy")
    total, steps = 0.0, 0
    model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(...).logits
        preds = torch.argmax(logits, dim=-1)
        total += metric.compute(...)["accuracy"]
        steps += 1
    return metric_name, total / max(steps, 1)
```

**特点**:
- **分离**: accuracy评估是独立函数
- **简洁**: 仅支持accuracy（不包含regression）
- **可扩展**: 通过 `TASK_CFG` 配置不同任务

### 3.2 src (acc_peak_time:360)

```python
@torch.no_grad()
def acc_peak_time(mdl, loader, device, task_name: str):
    if task_name == "stsb":
        metric = load_metric("pearsonr")
        metric_key = "pearsonr"
    else:
        metric = load_metric("accuracy")
        metric_key = "accuracy"
    # ... 同时测量 accuracy + perf
```

**特点**:
- **合并**: accuracy + perf 在一次遍历中完成
- **完整**: 支持分类(accuracy)和回归(pearsonr)
- **效率**: 减少数据集遍历次数

---

## 4. 关键差异总结

| 维度 | eval_encoder | src | 推荐 |
|------|--------------|-----|------|
| **Warmup** | ✅ 10步 | ❌ 无 | ✅ eval_encoder |
| **测量步数** | 固定50步 | 全数据集 | 🤔 取决于目标 |
| **多次运行** | ✅ 支持median | ❌ 单次 | ✅ eval_encoder |
| **吞吐量** | ✅ samples/s | ❌ 无 | ✅ eval_encoder |
| **Accuracy测量** | 分离函数 | 合并 | 🤔 取决于用途 |
| **任务支持** | accuracy only | accuracy + pearsonr | ✅ src (更全面) |
| **效率** | 2次遍历 | 1次遍历 | ✅ src (更快) |
| **可重复性** | ✅ 高 (median) | 🤷 中 | ✅ eval_encoder |

---

## 5. 实际测试对比

### 5.1 eval_encoder 配置
```bash
python eval_encoder/run_encoder_benchmark.py \
  --model_id textattack/bert-base-uncased-sst-2 \
  --method dense \
  --backend naive \
  --task sst2 \
  --batch_size 32 \
  --seq_len 128 \
  --warmup_steps 10 \
  --measure_steps 50 \
  --num_runs 3
```

**预期输出**:
- `metric_value`: 全validation集的accuracy
- `latency_ms`: 50步的median延迟
- `throughput_sps`: 50步的median吞吐量
- `peak_mem_mb`: 50步的median峰值内存

### 5.2 src 配置
```bash
cd src/encoders/BERTWhiting
python profile_svdllm.py
```

**实际输出** (来自之前的测试):
```
Data-aware (Whiting) | acc=0.8884 | peak=756.6 MiB | 310.4 ms/b
```

- `acc`: 0.8884 (全validation集)
- `peak`: 756.6 MiB
- `time`: 310.4 ms/batch

---

## 6. 测量差异分析

### 为什么峰值内存可能不同？

#### eval_encoder 可能**更低**的原因：
1. **固定步数**: 只跑50步，cache未满
2. **单个batch**: 可能不触发某些内存分配路径
3. **Warmup影响**: 某些allocation在warmup时已完成

#### src 可能**更高**的原因：
1. **全数据集**: 遍历所有batch，触发更多内存分配
2. **累积效应**: 某些中间tensor可能未及时释放
3. **计算accuracy**: 额外的CPU tensor分配（`preds.cpu()`）

### 为什么时间可能不同？

#### eval_encoder 可能**更快**的原因：
1. **Warmup**: CUDA kernel已预编译
2. **固定batch**: 避免不同sequence length的影响
3. **中位数**: 过滤掉异常值

#### src 可能**更慢**的原因：
1. **首次kernel编译**: 首批次包含编译开销
2. **全数据集**: 包含padding不同的batch
3. **Accuracy计算**: CPU同步和tensor拷贝开销

---

## 7. 实际测试验证

让我们运行eval_encoder来验证：

### 测试1: Dense baseline (与src对比)
```bash
python eval_encoder/run_encoder_benchmark.py \
  --model_id textattack/bert-base-uncased-sst-2 \
  --method dense \
  --task sst2 \
  --batch_size 32 \
  --seq_len 256 \
  --num_runs 3
```

**预期**: 应该与src的结果相近（考虑seq_len差异）

### 测试2: Whiten方法（如果支持）
```bash
python eval_encoder/run_encoder_benchmark.py \
  --model_id textattack/bert-base-uncased-sst-2 \
  --method drone \
  --rank 32 \
  --task sst2 \
  --batch_size 32 \
  --seq_len 256 \
  --num_runs 3
```

**预期**: 应该接近src的Whiten结果

---

## 8. 推荐的统一方案

### 方案A: 保持现状（推荐用于benchmark）
**适用场景**: 大规模对比实验，需要快速可重复结果

**使用**: eval_encoder
- ✅ Warmup + 固定步数
- ✅ 多次运行取median
- ✅ 吞吐量指标
- ❌ 不反映真实全数据集性能

### 方案B: 添加full-pass模式（推荐用于最终报告）
**适用场景**: 最终论文/报告，需要完整数据集结果

**修改**: 在 eval_encoder 中添加 `--full_pass` 选项
```python
if args.full_pass:
    # 类似src，遍历全数据集
    metric_val, peak, latency = evaluate_and_measure(model, loader, ...)
else:
    # 当前方式，固定步数
    metric_val = evaluate_task(...)
    latency, throughput, peak = measure_performance(...)
```

### 方案C: 统一接口
**目标**: 两种模式可互换

**实现**:
1. eval_encoder 添加 `--mode` 参数: `quick` (50步) vs `full` (全集)
2. src 添加 warmup 和 多次运行选项

---

## 9. 建议

### 立即可做：
1. ✅ 确认 eval_encoder 的默认参数（seq_len, batch_size）与src一致
2. ✅ 使用 eval_encoder 运行 dense baseline 验证
3. ✅ 对比两者的accuracy和latency

### 短期改进：
1. 在 eval_encoder 中添加 `--full_validation` flag
   - 如果设置，遍历全validation集
   - 保持warmup和median功能

2. 在 src/profile_svdllm.py 中添加 warmup
   ```python
   # Warmup
   for _ in range(10):
       model(...)
   torch.cuda.synchronize()
   torch.cuda.reset_peak_memory_stats()
   start = time.perf_counter()
   # ... existing code
   ```

### 长期统一：
1. 创建共享的 `eval_utils.py` 模块
2. 定义统一的 `BenchmarkConfig` dataclass
3. 所有脚本复用相同的测量逻辑

---

## 10. 结论

**当前状态**: 两个实现**测量逻辑相似但细节不同**

**主要差异**:
- eval_encoder: 更科学（warmup + median），快速可重复
- src: 更直接（全数据集），单次完整评估

**建议行动**:
1. **先验证**: 用 eval_encoder 跑 dense baseline，与src对比
2. **再统一**: 根据验证结果决定是否需要统一
3. **文档化**: 记录两种测量方式的适用场景

**不需要立即统一的理由**:
- 两者服务不同目的（快速实验 vs 完整评估）
- 只要测量逻辑一致（都正确重置内存、同步CUDA），结果应该可比
- 关键是**文档化差异**，让用户知道如何解释结果
