# 保有率参数使用指南

## 什么是保有率？

**保有率（Retention Rate）** 是一个更直观的压缩参数，表示压缩后保留原始模型维度的比例。

- **公式**: `retention = rank / hidden_size`
- **范围**: 0.0 ~ 1.0 (0% ~ 100%)
- **示例**: 对于 BERT-base (hidden_size=768)
  - `retention=0.5` → `rank=384` (保留 50% 维度)
  - `retention=0.3` → `rank=230` (保留 30% 维度)
  - `retention=0.7` → `rank=537` (保留 70% 维度)

## 为什么使用保有率？

### 优势对比

| 方式 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **指定 rank** | 精确控制秩大小 | 不同模型需要调整数值 | 精细实验、已知最优秩 |
| **指定保有率** | 跨模型一致性好 | 实际秩为整数近似 | 快速对比、跨模型实验 |

### 使用场景示例

#### 场景 1: 跨模型对比

```bash
# 使用固定 rank - 需要针对不同模型调整
# BERT-base (hidden=768)
RANK=384 bash eval_encoder/scripts/one_click_glue.sh

# BERT-large (hidden=1024) - 需要手动计算对应的 rank
RANK=512 MODEL_ID=bert-large-uncased bash eval_encoder/scripts/one_click_glue.sh

# 使用保有率 - 自动适配不同模型！
RETENTION=0.5 bash eval_encoder/scripts/one_click_glue.sh
RETENTION=0.5 MODEL_ID=bert-large-uncased bash eval_encoder/scripts/one_click_glue.sh
```

#### 场景 2: 系统性实验

```bash
# 测试不同压缩率的影响
for retention in 0.3 0.5 0.7; do
    RETENTION=$retention TASKS="sst2" bash eval_encoder/scripts/one_click_glue.sh
done
```

## 使用方法

### 方法 1: 通过环境变量（推荐）

```bash
# 使用保有率
RETENTION=0.5 bash eval_encoder/scripts/one_click_glue.sh

# 使用保有率 + 其他参数
RETENTION=0.3 METHOD=fwsvd TASKS="sst2 cola" \
    bash eval_encoder/scripts/one_click_glue.sh
```

### 方法 2: 直接调用 Python 脚本

```bash
# 使用保有率
python eval_encoder/glue_pipeline.py \
    --method fwsvd \
    --retention 0.5 \
    --tasks sst2 cola

# 使用固定 rank
python eval_encoder/glue_pipeline.py \
    --method fwsvd \
    --rank 300 \
    --tasks sst2 cola
```

### 方法 3: 使用 run_glue_benchmark.sh

```bash
bash eval_encoder/scripts/run_glue_benchmark.sh \
    --method fwsvd \
    --retention 0.5 \
    --tasks "sst2 cola"
```

## 参数优先级

**`--rank` 和 `--retention` 互斥，只能指定其中一个：**

```bash
# ✓ 正确：只指定 retention
RETENTION=0.5 bash eval_encoder/scripts/one_click_glue.sh

# ✓ 正确：只指定 rank
RANK=300 bash eval_encoder/scripts/one_click_glue.sh

# ✓ 正确：都不指定，使用默认 rank=300
bash eval_encoder/scripts/one_click_glue.sh

# ✗ 错误：同时指定两个会报错
RANK=300 RETENTION=0.5 bash eval_encoder/scripts/one_click_glue.sh
```

## 保有率与模型维度对照表

### BERT-base (hidden_size=768)

| 保有率 | 计算的 Rank | 参数压缩率 | 推荐用途 |
|--------|------------|-----------|---------|
| 0.9 | 691 | ~11% | 几乎无损压缩 |
| 0.7 | 537 | ~30% | 高质量压缩 |
| 0.5 | 384 | ~50% | 标准压缩 |
| 0.3 | 230 | ~70% | 激进压缩 |
| 0.2 | 153 | ~80% | 极限压缩 |
| 0.1 | 76 | ~90% | 研究性压缩 |

### BERT-large (hidden_size=1024)

| 保有率 | 计算的 Rank | 参数压缩率 |
|--------|------------|-----------|
| 0.9 | 921 | ~11% |
| 0.7 | 716 | ~30% |
| 0.5 | 512 | ~50% |
| 0.3 | 307 | ~70% |
| 0.2 | 204 | ~80% |
| 0.1 | 102 | ~90% |

### RoBERTa-base (hidden_size=768)

与 BERT-base 相同

## 实验建议

### 快速探索不同压缩率

```bash
#!/bin/bash
# 测试不同保有率对性能的影响

for retention in 0.1 0.2 0.3 0.5 0.7; do
    echo "Testing retention: $retention"
    RETENTION=$retention \
    METHOD=fwsvd \
    TASKS="sst2" \
    NUM_EPOCHS=3 \
        bash eval_encoder/scripts/one_click_glue.sh
done
```

### 不同压缩方法对比（相同保有率）

```bash
#!/bin/bash
# 在相同保有率下对比不同压缩方法

RETENTION=0.5
TASKS="sst2 cola mrpc"

for method in svd fwsvd drone; do
    echo "Testing method: $method"
    RETENTION=$RETENTION \
    METHOD=$method \
    TASKS="$TASKS" \
        bash eval_encoder/scripts/one_click_glue.sh
done
```

### 完整消融实验

```bash
#!/bin/bash
# 系统性测试：3种方法 × 4种保有率 × 3个任务

TASKS="sst2 cola mrpc"

for method in svd fwsvd drone; do
    for retention in 0.3 0.5 0.7 0.9; do
        echo "Running: $method @ retention=$retention"
        RETENTION=$retention \
        METHOD=$method \
        TASKS="$TASKS" \
        NUM_EPOCHS=3 \
            bash eval_encoder/scripts/one_click_glue.sh
    done
done
```

## 注意事项

### 1. 整数近似

由于 rank 必须是整数，实际的保有率会有小幅偏差：

```
retention=0.33 → rank=253 → actual_retention=0.3294 (略低)
retention=0.34 → rank=261 → actual_retention=0.3398 (略高)
```

### 2. 与 AdaSVD 的区别

- **保有率参数**: 固定每层使用相同的秩
- **AdaSVD budget**: 自适应为每层分配不同的秩

```bash
# 固定保有率（所有层 rank 相同）
RETENTION=0.5 METHOD=fwsvd bash eval_encoder/scripts/one_click_glue.sh

# AdaSVD 自适应（每层 rank 不同）
# 推荐使用 BUDGET=0.5 或 0.6（0.3太激进，准确率会大幅下降）
METHOD=adasvd BUDGET=0.5 bash eval_encoder/scripts/one_click_glue.sh
```

### 3. 模型检查点命名

**注意：无论使用 rank 还是 retention，模型名称格式都相同：**

```
# 使用 rank=300
eval_encoder/models/fwsvd_r300_naive/

# 使用 retention=0.5 (计算得到 rank=384)
eval_encoder/models/fwsvd_r384_naive/
                    ^^^^
                    实际使用的rank
```

retention 参数在压缩时会自动转换为 rank，模型目录名称只包含最终的 rank 值。

### 4. 不支持 dense 和 adasvd

保有率参数仅适用于基于 SVD 的方法：

- ✓ 支持: `svd`, `fwsvd`, `drone`
- ✗ 不支持: `dense` (无压缩), `adasvd` (使用 budget)

## 常见问题

### Q1: 如何选择合适的保有率？

**建议起点**:
- **首次尝试**: `retention=0.5` (平衡点)
- **高质量需求**: `retention=0.7` (轻度压缩)
- **激进压缩**: `retention=0.3` (大幅压缩)

**逐步调整**:
```bash
# 从 0.5 开始
RETENTION=0.5 TASKS="sst2" bash eval_encoder/scripts/one_click_glue.sh

# 如果精度损失小，尝试更低
RETENTION=0.3 TASKS="sst2" bash eval_encoder/scripts/one_click_glue.sh

# 如果精度损失大，尝试更高
RETENTION=0.7 TASKS="sst2" bash eval_encoder/scripts/one_click_glue.sh
```

### Q2: 保有率和 budget 有什么区别？

| 参数 | 适用方法 | 分配策略 | 灵活性 |
|------|---------|---------|--------|
| **retention** | svd/fwsvd/drone | 每层相同秩 | 简单统一 |
| **budget** | adasvd | 每层不同秩 | 自适应优化 |

### Q3: 如何验证实际使用的 rank？

脚本会在开始时打印计算结果：

```
[retention] Model: bert-base-uncased
[retention] Hidden size: 768
[retention] Retention rate: 50.00%
[retention] Calculated rank: 384
```

也可以查看模型文件夹名称：
```bash
ls eval_encoder/models/
# 输出: fwsvd_r384_naive
#              ^^^
#              实际rank (由retention=0.5计算得到)
```

### Q4: 可以在已有模型上修改保有率吗？

不可以。保有率在**压缩时确定**，需要重新压缩：

```bash
# 重新压缩使用新的保有率
RETENTION=0.3 bash eval_encoder/scripts/one_click_glue.sh
```

## 高级技巧

### 1. 二分搜索最优保有率

```bash
#!/bin/bash
# 找到满足精度阈值的最小保有率

THRESHOLD=0.90  # 最低可接受精度
LOW=0.1
HIGH=1.0

while (( $(echo "$HIGH - $LOW > 0.05" | bc -l) )); do
    MID=$(echo "scale=2; ($LOW + $HIGH) / 2" | bc)
    echo "Testing retention: $MID"

    RETENTION=$MID TASKS="sst2" bash eval_encoder/scripts/one_click_glue.sh

    # 检查结果（需要手动或自动解析）
    # 如果 accuracy >= THRESHOLD: HIGH=$MID
    # 否则: LOW=$MID
done
```

### 2. 可视化保有率-精度曲线

```python
# analyze_retention.py
import json
import matplotlib.pyplot as plt

retentions = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
accuracies = []

for ret in retentions:
    # 读取对应的结果文件
    result_file = f"eval_encoder/glue_results/glue_results_fwsvd_ret{int(ret*100)}_*.json"
    with open(result_file) as f:
        data = json.load(f)
        acc = data['results'][0]['final_results']['accuracy']
        accuracies.append(acc)

plt.plot(retentions, accuracies, marker='o')
plt.xlabel('Retention Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Retention Rate')
plt.grid(True)
plt.savefig('retention_accuracy_curve.png')
```

## 总结

保有率参数提供了更直观、跨模型一致的压缩控制方式：

✅ **推荐使用保有率的情况**:
- 跨模型对比实验
- 系统性测试不同压缩率
- 对 rank 数值不敏感的探索

✅ **推荐使用 rank 的情况**:
- 已知某个特定 rank 效果最好
- 需要精确复现某个实验
- 与已有工作对齐

**一般建议**: 先用保有率快速探索，找到合适的范围后，再用 rank 精细调优。
