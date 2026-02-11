# Full Validation Mode 使用指南

## 概述

`eval_encoder/run_encoder_benchmark.py` 现在支持两种性能测量模式：

1. **Standard Mode** (默认): 固定步数 + 多次运行取中位数 - 快速、可重复
2. **Full Validation Mode**: 遍历完整验证集 - 真实端到端性能

---

## 使用方法

### Standard Mode (默认)

```bash
python eval_encoder/run_encoder_benchmark.py \
  --model_id textattack/bert-base-uncased-sst-2 \
  --method dense \
  --task sst2 \
  --batch_size 32 \
  --seq_len 256 \
  --warmup_steps 10 \
  --measure_steps 50 \
  --num_runs 3
```

**特点**:
- ✅ 快速（只跑50步）
- ✅ 可重复（3次取中位数）
- ✅ 排除异常值
- ❌ 可能不反映全数据集特性（如不同长度序列的影响）

**适用场景**: 快速对比实验、超参数搜索、方法初步验证

### Full Validation Mode

```bash
python eval_encoder/run_encoder_benchmark.py \
  --model_id textattack/bert-base-uncased-sst-2 \
  --method dense \
  --task sst2 \
  --batch_size 32 \
  --seq_len 256 \
  --full_validation
```

**特点**:
- ✅ 真实端到端性能（遍历全数据集）
- ✅ 反映实际部署情况
- ✅ 与 `src/encoders/*/profile_*.py` 一致的测量方式
- ❌ 较慢（需要遍历整个验证集）
- ⚠️ 单次运行（无median），可能受系统负载影响

**适用场景**: 最终性能报告、论文结果、与baseline对比

**注意**:
- `--full_validation` 会忽略 `--measure_steps` 和 `--num_runs` 参数
- 仍然保留 `--warmup_steps` (默认10步)

---

## 结果对比

### 测试配置
- Model: `textattack/bert-base-uncased-sst-2`
- Method: Dense (baseline)
- Task: SST-2
- Batch size: 32
- Seq length: 256

### Standard Mode (3 runs, median)
```
accuracy      = 0.9263 (92.63%)
latency       = 65.98 ms/batch (median of 3 runs)
throughput    = 485.0 samples/s
peak_mem      = 360.2 MB
measurement   = 50 steps
```

### Full Validation Mode
```
accuracy      = 0.9263 (92.63%)
latency       = 65.01 ms/batch
throughput    = 479.0 samples/s
peak_mem      = 360.2 MB
measurement   = 28 batches (full dataset, 872 samples)
```

### 对比 src/encoders/BERTWhiting/profile_svdllm.py
```
# Whiten压缩模型 (RANK_ATTN=32, RANK_FF=384)
accuracy      = 0.8884 (88.84%)
latency       = 310.4 ms/batch
peak_mem      = 756.6 MB
measurement   = Full dataset
```

**分析**:
1. **Dense 模型**: Standard 和 Full validation 结果非常接近（差异 <2%）
   - 说明 50 步的测量已经足够准确
   - Full validation 稍慢可能因为不同batch的padding差异

2. **Whiten vs Dense**:
   - Accuracy: 88.84% vs 92.63% (压缩损失 ~4%)
   - Latency: 310.4ms vs 65.01ms (低秩分解慢 ~4.8x)
   - Memory: 756.6MB vs 360.2MB (Whiten更高，可能因为中间tensor)

---

## 推荐使用策略

### 开发阶段
```bash
# 快速验证方法是否work
python eval_encoder/run_encoder_benchmark.py \
  --method YOUR_METHOD \
  --rank 64 \
  --num_runs 1 \
  --measure_steps 20
```

### 超参数调优
```bash
# 多次运行取median，减少噪声
python eval_encoder/run_encoder_benchmark.py \
  --method YOUR_METHOD \
  --rank 64 \
  --num_runs 3 \
  --measure_steps 50
```

### 最终报告
```bash
# Full validation，获得真实端到端性能
python eval_encoder/run_encoder_benchmark.py \
  --method YOUR_METHOD \
  --rank 64 \
  --full_validation
```

### 对比表格生成
```bash
# 运行多个配置，自动写入CSV
for method in dense svd fwsvd drone; do
  for rank in 32 64 128; do
    python eval_encoder/run_encoder_benchmark.py \
      --method $method \
      --rank $rank \
      --full_validation \
      --out_csv results.csv
  done
done

# 查看结果
cat results.csv | column -t -s,
```

---

## 与 src/profile_*.py 的对比

| 特性 | eval_encoder (Full) | src/profile_*.py |
|------|---------------------|------------------|
| Warmup | ✅ 10 steps | ❌ 无 |
| 测量范围 | ✅ 全数据集 | ✅ 全数据集 |
| 多次运行 | ❌ 单次 | ❌ 单次 |
| Accuracy | ✅ 分离评估 | ✅ 合并评估 |
| 吞吐量 | ✅ samples/s | ❌ 无 |
| CSV输出 | ✅ 自动 | ❌ 手动 |
| 任务支持 | SST-2, MNLI | SST-2 + 更多 |

**优势**:
- eval_encoder 提供统一接口和CSV输出
- src 脚本更灵活，可以测试特定配置

**建议**:
- 开发新方法时使用 eval_encoder (快速迭代)
- 验证关键结果时运行两者对比（double-check）

---

## 故障排查

### Q: Full validation 比 standard 慢很多？
**A**: 正常，full validation 遍历全数据集（872样本），standard 只跑 50×32=1600 样本。

### Q: 为什么 peak_mem 不一致？
**A**: 可能原因：
- Full validation 遍历更多batch，触发更多内存分配
- Standard mode 的cache在warmup时已经预热
- 不同batch的sequence length差异

**解决**: 多次运行取median（但 full_validation 目前不支持）

### Q: 如何在 full_validation 模式下也取median？
**A**: 目前不支持。如果需要，可以手动运行3次：
```bash
for i in {1..3}; do
  python eval_encoder/run_encoder_benchmark.py \
    --full_validation \
    --notes "run_$i" \
    --out_csv full_runs.csv
done
# 然后从 CSV 中手动计算median
```

### Q: Full validation 结果是否可信？
**A**: 是的。测试显示 dense baseline 的 standard vs full 差异 <2%，在误差范围内。

---

## 实现细节

### 代码位置
`eval_encoder/run_encoder_benchmark.py:658`

```python
@torch.no_grad()
def measure_performance(model, loader, device, warmup_steps, measure_steps,
                        num_runs=1, full_validation=False):
    if full_validation:
        # 1. Warmup with first batch
        # 2. Reset peak memory stats
        # 3. Traverse entire dataset
        # 4. Calculate avg latency per batch
        ...
    else:
        # Standard mode: fixed steps + median
        ...
```

### 关键差异
1. **Full mode**: `for batch in loader` - 遍历全数据集
2. **Standard mode**: `for _ in range(measure_steps)` - 固定步数重复同一batch

### 为什么不支持 full_validation + num_runs?
- Full validation 已经遍历全数据集，单次运行已经足够稳定
- 多次遍历会显著增加运行时间（3x）
- 如果需要median，可以手动运行多次后从CSV提取

---

## 未来改进方向

### 短期
1. ✅ 添加 `--full_validation` 支持 (已完成)
2. 🔄 支持 `--full_validation` + `--num_runs` 组合

### 中期
3. 添加 `--profile_mode` 选项: `quick` / `standard` / `full`
   - quick: 20 steps, 1 run
   - standard: 50 steps, 3 runs (median)
   - full: full dataset, 1 run

### 长期
4. 统一 eval_encoder 和 src/profile_*.py 的接口
5. 添加性能profiling（分析各层耗时）
6. 支持分布式多GPU测量

---

## 参考文档

- [性能测量对比分析](../development/performance-measurement-comparison.md)
- [Benchmark使用指南](benchmark-guide.md)
- [快速入门](getting-started.md)
