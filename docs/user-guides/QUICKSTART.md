# 一键式 GLUE 评测 - 快速开始

## 🚀 超快速开始（3 行命令）

```bash
# 1. 测试环境
bash eval_encoder/scripts/test_setup.sh

# 2. 运行评测（默认使用任务特定模型）
bash eval_encoder/scripts/one_click_glue.sh

# 3. 查看结果
cat eval_encoder/glue_results/glue_results_*.json | python -m json.tool
```

## 📌 模型选择

### 默认：任务特定模型（推荐）

**默认使用 HuggingFace 上已经微调好的模型**（如 `textattack/bert-base-uncased-SST-2`）

```bash
# 默认就是任务特定模型，无需额外参数
bash eval_encoder/scripts/one_click_glue.sh
```

**优势**:
- ✅ 更高的初始准确率（93% vs 50% for SST-2）
- ✅ 压缩后保留更多性能
- ✅ 微调后恢复更好（92.5% vs 91.2%）

### 可选：基础模型

如果想使用基础模型进行跨任务实验：

```bash
# 使用基础模型（如 bert-base-uncased）
USE_TASK_MODELS=false bash eval_encoder/scripts/one_click_glue.sh
```

详见: **`eval_encoder/PRETRAINED_MODELS.md`**

---

## 📋 环境要求

- Python 3.8+
- CUDA 11.0+
- PyTorch 1.13+
- Transformers, Datasets, Evaluate
- NVIDIA GPU (至少 8GB 显存)

**安装依赖：**
```bash
pip install torch transformers datasets evaluate scikit-learn scipy tqdm
```

---

## ⚙️ 自定义配置

### 方式 1: 环境变量（推荐）

```bash
# 快速测试（约 1 小时）
TASKS="sst2" bash eval_encoder/scripts/one_click_glue.sh

# 标准评测（约 6 小时）
TASKS="cola sst2 mrpc qnli rte stsb" bash eval_encoder/scripts/one_click_glue.sh

# 完整 GLUE（约 16 小时）
bash eval_encoder/scripts/one_click_glue.sh  # 默认跑全部

# 自定义压缩方法和秩
METHOD=fwsvd RANK=256 TASKS="sst2 cola" bash eval_encoder/scripts/one_click_glue.sh

# 🆕 使用保有率参数（更直观！）
# 保有率 0.5 = 保留 50% 维度 = rank=384 (BERT-base)
RETENTION=0.5 METHOD=fwsvd TASKS="sst2" bash eval_encoder/scripts/one_click_glue.sh

# 测试不同保有率
RETENTION=0.3 TASKS="sst2" bash eval_encoder/scripts/one_click_glue.sh  # 30% 激进压缩
RETENTION=0.5 TASKS="sst2" bash eval_encoder/scripts/one_click_glue.sh  # 50% 标准压缩
RETENTION=0.7 TASKS="sst2" bash eval_encoder/scripts/one_click_glue.sh  # 70% 轻度压缩
```

### 方式 2: 修改脚本配置

编辑 `eval_encoder/scripts/one_click_glue.sh` 文件：

```bash
# 在文件顶部修改这些变量
METHOD="fwsvd"                    # 压缩方法
RANK=300                          # SVD 秩
TASKS="cola sst2 mrpc qqp mnli qnli rte stsb"  # 任务列表
NUM_EPOCHS=3                      # 训练轮数
BATCH_SIZE=16                     # 批大小
```

---

## 📊 支持的任务

| 任务 | 时间估计 | 数据集大小 | 评估指标 |
|------|---------|-----------|---------|
| CoLA | ~10min | 8.5k | Matthews相关系数 |
| SST-2 | ~1h | 67k | 准确率 |
| MRPC | ~6min | 3.7k | F1 |
| QQP | ~6h | 364k | F1 |
| MNLI | ~7h | 393k | 准确率 |
| QNLI | ~2h | 105k | 准确率 |
| RTE | ~3min | 2.5k | 准确率 |
| STS-B | ~10min | 5.7k | Pearson相关 |

---

## 🎯 使用场景

### 场景 1: 调试/测试

```bash
# 只跑最小任务（3分钟）
TASKS="rte" NUM_EPOCHS=1 bash eval_encoder/scripts/one_click_glue.sh
```

### 场景 2: 快速验证（推荐首次使用）

```bash
# 跑小任务集合（1.5小时）
TASKS="cola sst2 mrpc rte" bash eval_encoder/scripts/one_click_glue.sh
```

### 场景 3: 标准论文实验

```bash
# 跑 6 个常用任务（6小时）
TASKS="cola sst2 mrpc qnli rte stsb" bash eval_encoder/scripts/one_click_glue.sh
```

### 场景 4: 完整评测

```bash
# 跑全部 8 个任务（16小时，建议 overnight）
bash eval_encoder/scripts/one_click_glue.sh
```

### 场景 5: 对比实验

```bash
# 测试不同方法
for method in svd fwsvd drone; do
    METHOD=$method RANK=300 TASKS="sst2 cola" \
        bash eval_encoder/scripts/one_click_glue.sh
done

# 测试不同秩
for rank in 128 256 400; do
    METHOD=fwsvd RANK=$rank TASKS="sst2" \
        bash eval_encoder/scripts/one_click_glue.sh
done
```

---

## 💡 常见问题

### Q1: 内存不够？

```bash
# 降低 batch size
BATCH_SIZE=8 bash eval_encoder/scripts/one_click_glue.sh
# 或
BATCH_SIZE=4 bash eval_encoder/scripts/one_click_glue.sh
```

### Q2: 训练太慢？

```bash
# 减少 epochs
NUM_EPOCHS=1 bash eval_encoder/scripts/one_click_glue.sh

# 或只跑小任务
TASKS="cola mrpc rte" bash eval_encoder/scripts/one_click_glue.sh
```

### Q3: 温度过高？

```bash
# 降低 batch size 减少发热
BATCH_SIZE=4 bash eval_encoder/scripts/one_click_glue.sh

# 或在另一个终端监控
watch -n 5 nvidia-smi
```

### Q4: 网络问题？

```bash
# 使用 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
bash eval_encoder/scripts/one_click_glue.sh
```

---

## 📁 输出文件

```
eval_encoder/
├── glue_results/
│   ├── glue_results_fwsvd_20260213_123456.json  # 完整结果
│   └── logs/
│       └── fwsvd_r300_20260213_123456.log       # 完整日志
└── models/
    └── fwsvd_r300_naive/                        # 压缩模型
```

**查看结果：**
```bash
# 格式化显示
cat eval_encoder/glue_results/glue_results_*.json | python -m json.tool

# 查看日志
tail -100 eval_encoder/glue_results/logs/*.log

# 提取关键指标
cat eval_encoder/glue_results/glue_results_*.json | \
    python -c "
import json, sys
d = json.load(sys.stdin)
print('Task      | Metric                | Initial | Final')
print('----------|----------------------|---------|-------')
for r in d['results']:
    print(f\"{r['task']:10} | {r['best_metric']:20} | {list(r['initial_results'].values())[0]:.4f}  | {r['best_value']:.4f}\")
"
```

---

## 🔧 后台运行

### 使用 nohup

```bash
nohup bash eval_encoder/scripts/one_click_glue.sh \
    > /tmp/glue_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 查看进度
tail -f /tmp/glue_*.log
```

### 使用 tmux（推荐）

```bash
# 创建会话
tmux new -s glue

# 运行脚本
bash eval_encoder/scripts/one_click_glue.sh

# 断开会话（Ctrl+B, D）

# 重新连接
tmux attach -t glue
```

---

## 📦 部署到其他机器

### 步骤 1: 打包代码

```bash
# 在当前机器
tar -czf glue_eval.tar.gz lowrankarena/eval_encoder
```

### 步骤 2: 传输到目标机器

```bash
scp glue_eval.tar.gz user@remote:/path/to/
```

### 步骤 3: 在目标机器解压并运行

```bash
# SSH 登录
ssh user@remote

# 解压
tar -xzf glue_eval.tar.gz
cd lowrankarena

# 测试环境
bash eval_encoder/scripts/test_setup.sh

# 运行评测
bash eval_encoder/scripts/one_click_glue.sh
```

---

## 🆕 保有率参数（推荐使用）

**什么是保有率？** 更直观的压缩参数，表示保留原始维度的比例。

```bash
# 传统方式：指定 rank（需要根据模型调整）
RANK=384 bash eval_encoder/scripts/one_click_glue.sh

# 🆕 新方式：指定保有率（自动适配不同模型）
RETENTION=0.5 bash eval_encoder/scripts/one_click_glue.sh
```

**保有率对照表 (BERT-base, hidden=768)**:

| 保有率 | 实际 Rank | 压缩率 | 推荐用途 |
|--------|----------|-------|---------|
| 0.9 | 691 | 11% | 几乎无损 |
| 0.7 | 537 | 30% | 高质量压缩 |
| 0.5 | 384 | 50% | **标准压缩（推荐）** |
| 0.3 | 230 | 70% | 激进压缩 |
| 0.1 | 76 | 90% | 极限压缩 |

**快速实验**:
```bash
# 测试不同压缩率
for ret in 0.3 0.5 0.7; do
    RETENTION=$ret TASKS="sst2" bash eval_encoder/scripts/one_click_glue.sh
done
```

详见: **`eval_encoder/RETENTION_GUIDE.md`**

---

## 📚 详细文档

- **保有率使用指南**: `eval_encoder/RETENTION_GUIDE.md` 🆕
- **完整部署指南**: `eval_encoder/DEPLOYMENT.md`
- **GLUE 评测详解**: `eval_encoder/GLUE_BENCHMARK.md`
- **API 文档**: `eval_encoder/glue_pipeline.py` (查看 docstring)

---

## ✅ 检查清单

运行前确保：

- [ ] 已安装 CUDA 和 GPU 驱动
- [ ] 已安装 Python 和必要包
- [ ] GPU 显存至少 8GB
- [ ] 磁盘空间至少 20GB
- [ ] 网络可访问 HuggingFace Hub
- [ ] 脚本有执行权限

运行环境测试验证：
```bash
bash eval_encoder/scripts/test_setup.sh
```

---

**祝评测顺利！🎉**

有问题请查看 `eval_encoder/DEPLOYMENT.md` 或日志文件。
