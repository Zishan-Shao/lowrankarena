# GLUE 评测指标详解

## 各任务使用的评估指标

| 任务 | 主要指标 | 次要指标 | 指标范围 | 说明 |
|------|---------|---------|---------|------|
| **CoLA** | Matthews Correlation | - | -1.0 ~ 1.0 | 相关系数，0表示随机 |
| **SST-2** | Accuracy | - | 0.0 ~ 1.0 | 准确率 |
| **MRPC** | F1 | Accuracy | 0.0 ~ 1.0 | F1为主，准确率为辅 |
| **QQP** | F1 | Accuracy | 0.0 ~ 1.0 | F1为主，准确率为辅 |
| **MNLI** | Accuracy (matched) | Accuracy (mismatched) | 0.0 ~ 1.0 | 两个验证集 |
| **QNLI** | Accuracy | - | 0.0 ~ 1.0 | 准确率 |
| **RTE** | Accuracy | - | 0.0 ~ 1.0 | 准确率 |
| **STS-B** | Pearson Correlation | Spearman Correlation | -1.0 ~ 1.0 | 回归任务，相关系数 |

## 详细说明

### 1. CoLA (Corpus of Linguistic Acceptability)
**任务**: 判断句子语法是否可接受（二分类）

**指标**: Matthews Correlation Coefficient (MCC)
```
MCC = (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**为什么用 MCC？**
- 数据集高度不平衡（可接受句子多，不可接受的少）
- MCC 对不平衡数据更敏感
- 比准确率更能反映真实性能

**解读**:
- **1.0**: 完美预测
- **0.0**: 随机猜测
- **-1.0**: 完全相反
- **>0.5**: 优秀
- **0.3~0.5**: 良好
- **<0.3**: 较差

**示例输出**:
```json
{
  "matthews_correlation": 0.5234
}
```

---

### 2. SST-2 (Stanford Sentiment Treebank)
**任务**: 电影评论情感分析（二分类：正面/负面）

**指标**: Accuracy (准确率)
```
Accuracy = 正确预测数 / 总样本数
```

**解读**:
- **>0.93**: SOTA 水平
- **0.90~0.93**: 优秀
- **0.85~0.90**: 良好
- **<0.85**: 需要改进

**示例输出**:
```json
{
  "accuracy": 0.9150
}
```

---

### 3. MRPC (Microsoft Research Paraphrase Corpus)
**任务**: 判断两个句子是否为释义关系（二分类）

**主要指标**: F1 Score
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**次要指标**: Accuracy

**为什么用 F1？**
- 数据集不平衡（非释义对多）
- F1 同时考虑精确率和召回率
- 对少数类（释义对）更敏感

**解读**:
- **>0.90**: SOTA 水平
- **0.87~0.90**: 优秀
- **0.80~0.87**: 良好
- **<0.80**: 需要改进

**示例输出**:
```json
{
  "accuracy": 0.8456,
  "f1": 0.8734
}
```
*注：官方排行榜使用 F1 作为主要指标*

---

### 4. QQP (Quora Question Pairs)
**任务**: 判断两个问题是否重复（二分类）

**主要指标**: F1 Score

**次要指标**: Accuracy

**解读**:
- **>0.72**: SOTA 水平
- **0.70~0.72**: 优秀
- **0.65~0.70**: 良好
- **<0.65**: 需要改进

**示例输出**:
```json
{
  "accuracy": 0.8912,
  "f1": 0.7089
}
```

---

### 5. MNLI (Multi-Genre Natural Language Inference)
**任务**: 自然语言推理（三分类：蕴含/中性/矛盾）

**指标**: Accuracy (两个验证集)
- **Matched**: 与训练数据同领域
- **Mismatched**: 与训练数据不同领域

**解读**:
- **>0.86**: SOTA 水平
- **0.84~0.86**: 优秀
- **0.80~0.84**: 良好
- **<0.80**: 需要改进

**示例输出**:
```json
{
  "accuracy": 0.8423
}
```
*注：我们的脚本使用 validation_matched*

---

### 6. QNLI (Question Natural Language Inference)
**任务**: 问答句对推理（二分类：蕴含/不蕴含）

**指标**: Accuracy

**解读**:
- **>0.92**: SOTA 水平
- **0.90~0.92**: 优秀
- **0.85~0.90**: 良好
- **<0.85**: 需要改进

**示例输出**:
```json
{
  "accuracy": 0.9012
}
```

---

### 7. RTE (Recognizing Textual Entailment)
**任务**: 文本蕴含识别（二分类：蕴含/不蕴含）

**指标**: Accuracy

**特点**: 数据集很小（2.5k样本），结果波动大

**解读**:
- **>0.70**: SOTA 水平
- **0.66~0.70**: 优秀
- **0.60~0.66**: 良好
- **<0.60**: 需要改进

**示例输出**:
```json
{
  "accuracy": 0.6534
}
```

---

### 8. STS-B (Semantic Textual Similarity Benchmark)
**任务**: 语义相似度评分（回归任务，0-5分）

**主要指标**: Pearson Correlation Coefficient
```
Pearson r = cov(X,Y) / (σ_X × σ_Y)
```

**次要指标**: Spearman Correlation Coefficient

**为什么用相关系数？**
- 回归任务，预测连续值
- 相关系数衡量预测与真实值的线性关系
- 不受量纲影响

**解读**:
- **>0.90**: SOTA 水平
- **0.85~0.90**: 优秀
- **0.80~0.85**: 良好
- **<0.80**: 需要改进

**示例输出**:
```json
{
  "pearson": 0.8634,
  "spearmanr": 0.8512
}
```

---

## GLUE 总分计算

GLUE 官方排行榜使用所有任务的平均分：

```
GLUE Score = (CoLA_MCC + SST-2_Acc + MRPC_F1 + QQP_F1 + MNLI_Acc_m
             + MNLI_Acc_mm + QNLI_Acc + RTE_Acc + STS-B_Pearson) / 9
```

**注意**:
- CoLA 使用 MCC（范围 -1~1），需要转换为 0~1
- MRPC 和 QQP 使用 F1
- STS-B 使用 Pearson 相关系数（范围 -1~1），需要转换为 0~1

---

## 我们的脚本输出格式

### 单任务结果

```json
{
  "task": "sst2",
  "initial_results": {
    "accuracy": 0.8544
  },
  "final_results": {
    "accuracy": 0.9150
  },
  "best_metric": "accuracy",
  "best_value": 0.9150
}
```

### 所有任务汇总

```json
{
  "timestamp": "20260213_123456",
  "config": {...},
  "results": [
    {
      "task": "cola",
      "best_metric": "matthews_correlation",
      "best_value": 0.5234
    },
    {
      "task": "sst2",
      "best_metric": "accuracy",
      "best_value": 0.9150
    },
    ...
  ]
}
```

---

## 如何查看详细指标

### 方法 1: 读取 JSON 文件

```bash
cat eval_encoder/glue_results/glue_results_*.json | python -m json.tool
```

### 方法 2: 提取关键指标

```bash
cat eval_encoder/glue_results/glue_results_*.json | python -c "
import json, sys
data = json.load(sys.stdin)

print('Task      | Metric               | Initial | Final  | Δ')
print('----------|----------------------|---------|--------|-------')

for r in data['results']:
    task = r['task']
    metric = r['best_metric']
    initial = list(r['initial_results'].values())[0]
    final = r['best_value']
    delta = final - initial

    print(f'{task:10} | {metric:20} | {initial:.4f}  | {final:.4f} | {delta:+.4f}')
"
```

### 方法 3: 计算 GLUE 总分

```bash
cat eval_encoder/glue_results/glue_results_*.json | python -c "
import json, sys
data = json.load(sys.stdin)

scores = []
for r in data['results']:
    metric = r['best_value']
    # MCC 和 Pearson 已经是 -1~1，转换为 0~1
    if r['best_metric'] in ['matthews_correlation', 'pearson']:
        metric = (metric + 1) / 2
    scores.append(metric)

glue_score = sum(scores) / len(scores)
print(f'GLUE Score: {glue_score:.4f}')
"
```

---

## 参考基准

### BERT-base-uncased (原始模型)

| 任务 | 指标 | 分数 | 备注 |
|------|------|------|------|
| CoLA | MCC | 52.1 | |
| SST-2 | Acc | 93.5 | |
| MRPC | F1 | 88.9 | Acc: 84.8 |
| QQP | F1 | 71.2 | Acc: 89.2 |
| MNLI | Acc | 84.6 | matched |
| QNLI | Acc | 90.5 | |
| RTE | Acc | 66.4 | 数据少，波动大 |
| STS-B | Pearson | 85.8 | Spearman: 85.5 |
| **GLUE** | **Average** | **78.3** | |

### 压缩模型预期性能

#### FWSVD r=300 (retention=0.4)

| 任务 | 压缩后 | 微调后 | 恢复率 |
|------|--------|--------|--------|
| CoLA | 48-50 | 50-52 | ~96% |
| SST-2 | 85-88 | 91-93 | ~97% |
| MRPC | 82-85 | 86-89 | ~97% |
| QQP | 66-69 | 70-72 | ~99% |
| MNLI | 78-81 | 82-85 | ~98% |
| QNLI | 84-87 | 88-91 | ~98% |
| RTE | 56-60 | 63-67 | ~95% |
| STS-B | 78-82 | 83-86 | ~98% |

---

## 总结

我们的脚本会自动：
1. ✅ 使用正确的评估指标
2. ✅ 计算初始和最终性能
3. ✅ 保存完整的评估结果
4. ✅ 输出人类可读的摘要

查看完整结果：
```bash
cat eval_encoder/glue_results/glue_results_*.json | python -m json.tool
```
