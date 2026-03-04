# 支持的 GLUE 任务

根据 `eval_encoder/glue_pipeline.py` (第58-163行)，我们支持以下 8 个 GLUE 基准任务：

## 任务列表

| # | 任务代码 | 全称 | 类型 | 标签数 | 主要评估指标 | 句子类型 |
|---|---------|------|------|--------|------------|---------|
| 1 | **cola** | Corpus of Linguistic Acceptability | 分类 | 2 | Matthews Correlation | 单句 |
| 2 | **sst2** | Stanford Sentiment Treebank | 分类 | 2 | Accuracy | 单句 |
| 3 | **mrpc** | Microsoft Research Paraphrase Corpus | 分类 | 2 | F1 Score | 句对 |
| 4 | **qqp** | Quora Question Pairs | 分类 | 2 | F1 Score | 句对 |
| 5 | **mnli** | Multi-Genre Natural Language Inference | 分类 | 3 | Accuracy | 句对 |
| 6 | **qnli** | Question Natural Language Inference | 分类 | 2 | Accuracy | 句对 |
| 7 | **rte** | Recognizing Textual Entailment | 分类 | 2 | Accuracy | 句对 |
| 8 | **stsb** | Semantic Textual Similarity Benchmark | 回归 | 1 | Pearson Correlation | 句对 |

## 任务详细信息

### 1. CoLA (Corpus of Linguistic Acceptability)
- **任务类型**: 单句分类
- **数据**: 判断句子的语法是否正确
- **输入字段**: `sentence`
- **评估指标**: Matthews Correlation (MCC)
- **标签数**: 2 (可接受/不可接受)
- **数据集大小**: 训练集 ~8.5K, 验证集 ~1K
- **预训练模型**:
  - `textattack/bert-base-uncased-CoLA`
  - `howey/bert-base-uncased-cola`

### 2. SST-2 (Stanford Sentiment Treebank)
- **任务类型**: 单句分类
- **数据**: 电影评论情感分析
- **输入字段**: `sentence`
- **评估指标**: Accuracy
- **标签数**: 2 (正面/负面)
- **数据集大小**: 训练集 ~67K, 验证集 ~872
- **预训练模型**:
  - `textattack/bert-base-uncased-SST-2`
  - `howey/bert-base-uncased-sst2`

### 3. MRPC (Microsoft Research Paraphrase Corpus)
- **任务类型**: 句对分类
- **数据**: 判断两个句子是否语义等价
- **输入字段**: `sentence1`, `sentence2`
- **评估指标**: F1 Score (同时报告 Accuracy)
- **标签数**: 2 (等价/不等价)
- **数据集大小**: 训练集 ~3.7K, 验证集 ~408
- **预训练模型**:
  - `textattack/bert-base-uncased-MRPC`
  - `howey/bert-base-uncased-mrpc`

### 4. QQP (Quora Question Pairs)
- **任务类型**: 句对分类
- **数据**: 判断两个问题是否语义相同
- **输入字段**: `question1`, `question2`
- **评估指标**: F1 Score (同时报告 Accuracy)
- **标签数**: 2 (重复/不重复)
- **数据集大小**: 训练集 ~364K, 验证集 ~40K
- **预训练模型**:
  - `textattack/bert-base-uncased-QQP`
  - `howey/bert-base-uncased-qqp`

### 5. MNLI (Multi-Genre Natural Language Inference)
- **任务类型**: 句对分类（三分类）
- **数据**: 判断前提和假设之间的推理关系
- **输入字段**: `premise`, `hypothesis`
- **评估指标**: Accuracy
- **标签数**: 3 (蕴含/中立/矛盾)
- **数据集大小**: 训练集 ~393K, 验证集 ~10K
- **特殊说明**: textattack 模型使用非标准标签映射，需要重映射 {0→2, 1→0, 2→1}
- **预训练模型**:
  - `textattack/bert-base-uncased-MNLI`
  - `howey/bert-base-uncased-mnli`

### 6. QNLI (Question Natural Language Inference)
- **任务类型**: 句对分类
- **数据**: 判断句子是否包含问题的答案
- **输入字段**: `question`, `sentence`
- **评估指标**: Accuracy
- **标签数**: 2 (蕴含/不蕴含)
- **数据集大小**: 训练集 ~105K, 验证集 ~5.5K
- **预训练模型**:
  - `textattack/bert-base-uncased-QNLI`
  - `howey/bert-base-uncased-qnli`

### 7. RTE (Recognizing Textual Entailment)
- **任务类型**: 句对分类
- **数据**: 判断前提是否蕴含假设
- **输入字段**: `sentence1`, `sentence2`
- **评估指标**: Accuracy
- **标签数**: 2 (蕴含/不蕴含)
- **数据集大小**: 训练集 ~2.5K, 验证集 ~277
- **预训练模型**:
  - `textattack/bert-base-uncased-RTE`
  - `howey/bert-base-uncased-rte`

### 8. STS-B (Semantic Textual Similarity Benchmark)
- **任务类型**: 句对回归
- **数据**: 预测两个句子的语义相似度 (0-5 分)
- **输入字段**: `sentence1`, `sentence2`
- **评估指标**: Pearson Correlation (同时报告 Spearman)
- **标签数**: 1 (连续值)
- **数据集大小**: 训练集 ~5.7K, 验证集 ~1.5K
- **预训练模型**:
  - `textattack/bert-base-uncased-STS-B`
  - `howey/bert-base-uncased-stsb`

## 平均指标说明

### G-Avg (GLUE Average)
- 所有 8 个任务主要指标的平均值
- 对于 MCC 和 Pearson (范围 -1~1)，先归一化到 0~1
- 公式: `(CoLA_norm + SST2 + MRPC + QQP + MNLI + QNLI + RTE + STSB_norm) / 8`

### A-Avg (Accuracy Average)
- 仅包含以 Accuracy 作为主要或次要指标的任务
- 包括任务: SST-2, MNLI, QNLI, RTE (主要 accuracy)
- MRPC, QQP (次要 accuracy，主要是 F1)
- 公式: `sum(accuracy_scores) / count(accuracy_tasks)`

## 使用方法

### 运行单个任务
```bash
python eval_encoder/glue_pipeline.py \
    --method fwsvd \
    --rank 256 \
    --backend flashsvd \
    --tasks sst2
```

### 运行所有任务
```bash
python eval_encoder/glue_pipeline.py \
    --method fwsvd \
    --rank 256 \
    --backend flashsvd \
    --tasks cola sst2 mrpc qqp mnli qnli rte stsb
```

### 使用任务特定的预训练模型
```bash
python eval_encoder/glue_pipeline.py \
    --use_task_models \
    --task_model_prefix textattack \
    --method fwsvd \
    --rank 256 \
    --tasks sst2
```

## 数据集大小排序

从小到大:
1. **RTE**: 2.5K (最小，适合快速测试)
2. **MRPC**: 3.7K
3. **STS-B**: 5.7K
4. **CoLA**: 8.5K
5. **SST-2**: 67K
6. **QNLI**: 105K
7. **QQP**: 364K (大型数据集)
8. **MNLI**: 393K (最大)

## 推荐测试顺序

### 快速验证 (小数据集)
```bash
--tasks rte mrpc cola
```

### 标准评估 (中等大小)
```bash
--tasks sst2 qnli
```

### 完整基准 (所有任务)
```bash
--tasks cola sst2 mrpc qqp mnli qnli rte stsb
```

## 相关文件

- **主脚本**: `eval_encoder/glue_pipeline.py`
- **模型加载**: `eval_encoder/load_compressed_model.py`
- **基准测试**: `eval_encoder/run_encoder_benchmark.py`
- **结果目录**: `eval_encoder/glue_results/`
- **模型保存**: `eval_encoder/models/{task}/{method}_r{rank}_{backend}/`

## 已知问题

1. **MNLI 标签映射**: textattack 模型需要特殊的标签重映射
2. **AdaSVD**: 部分配置可能产生异常结果（见实验结果）
3. **内存管理**: 大数据集（QQP, MNLI）需要足够的 GPU 内存

---

**文档版本**: 2026-02-17
**代码来源**: `eval_encoder/glue_pipeline.py:58-163`
