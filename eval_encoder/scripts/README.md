# Scripts 目录说明

本目录包含 eval_encoder 的核心脚本。

## 📁 目录结构

```
scripts/
├── README.md                         本文档
├── one_click_glue.sh                 一键 GLUE 评测（主要入口）
├── analyze_results.py                结果分析工具
├── generate_comparison_table.py      生成学术对比表格
├── test_setup.sh                     环境检测
├── utils/
│   └── cleanup.sh                    清理脚本
└── training/
    ├── compress_and_save.py          压缩并保存模型
    └── finetune_from_checkpoint.py   从checkpoint微调
```

## 🚀 核心脚本

### 1. one_click_glue.sh
**一键 GLUE 评测脚本**（主要入口）

完整流程：压缩 → 微调 → 评测 → 生成结果

```bash
# 基本使用
bash scripts/one_click_glue.sh

# 自定义配置
METHOD=fwsvd RETENTION=0.5 TASKS="sst2 cola" \
    bash scripts/one_click_glue.sh

# 使用 FlashSVD 后端
BACKEND=flashsvd RETENTION=0.5 \
    bash scripts/one_click_glue.sh
```

**支持的环境变量：**
- `METHOD`: 压缩方法 (svd, fwsvd, drone, adasvd)
- `RETENTION`: 保有率 (0.0-1.0)
- `RANK`: SVD 秩（与 RETENTION 互斥）
- `BACKEND`: 后端 (naive, flashsvd)
- `TASKS`: 任务列表（空格分隔）
- `NUM_EPOCHS`: 训练轮数
- `BATCH_SIZE`: 批大小
- `USE_TASK_MODELS`: 使用任务特定模型 (true/false)

## 📊 结果分析

### 2. analyze_results.py
**分析评测结果**

```bash
# 查看结果摘要
python scripts/analyze_results.py \
    eval_encoder/glue_results/glue_results_*.json

# 查看详细结果
python scripts/analyze_results.py \
    eval_encoder/glue_results/glue_results_*.json \
    --mode details

# 导出为 CSV
python scripts/analyze_results.py \
    eval_encoder/glue_results/glue_results_*.json \
    --mode csv \
    --output results.csv
```

### 3. generate_comparison_table.py
**生成学术对比表格**

```bash
# 生成 Markdown 表格
python scripts/generate_comparison_table.py \
    eval_encoder/glue_results/glue_results_*.json

# 生成 LaTeX 表格
python scripts/generate_comparison_table.py \
    eval_encoder/glue_results/glue_results_*.json \
    --format latex

# 生成 CSV 表格
python scripts/generate_comparison_table.py \
    eval_encoder/glue_results/glue_results_*.json \
    --format csv
```

## 🛠️ 工具脚本

### 4. test_setup.sh
**环境检测**

检查 Python、CUDA、GPU 和必要的依赖包。

```bash
bash scripts/test_setup.sh
```

### 5. utils/cleanup.sh
**清理脚本**

清理临时文件、旧日志和归档文件。

```bash
# 保守清理（约 200MB）
bash scripts/utils/cleanup.sh conservative

# 标准清理（约 500MB）
bash scripts/utils/cleanup.sh standard

# 深度清理（约 1.9GB，删除所有模型）
bash scripts/utils/cleanup.sh deep
```

## 📦 训练相关脚本

### 6. training/compress_and_save.py
**压缩并保存模型**

使用指定方法压缩模型并保存到本地。

```bash
python scripts/training/compress_and_save.py \
    --method fwsvd \
    --rank 300 \
    --output_dir eval_encoder/models/fwsvd_r300
```

### 7. training/finetune_from_checkpoint.py
**从 checkpoint 微调**

从保存的压缩模型继续微调。

```bash
python scripts/training/finetune_from_checkpoint.py \
    --checkpoint eval_encoder/models/fwsvd_r300 \
    --task sst2 \
    --num_epochs 3
```

## 🎯 常用工作流

### 工作流 1: 快速验证
```bash
# 单任务快速测试
RETENTION=0.5 TASKS="sst2" NUM_EPOCHS=1 \
    bash scripts/one_click_glue.sh
```

### 工作流 2: 标准评测
```bash
# 6任务标准评测 + FlashSVD 后端
BACKEND=flashsvd RETENTION=0.5 \
TASKS="cola sst2 mrpc qnli rte stsb" \
    bash scripts/one_click_glue.sh
```

### 工作流 3: 对比实验
```bash
# 测试不同保有率
for ret in 0.3 0.5 0.7; do
    RETENTION=$ret TASKS="sst2" \
        bash scripts/one_click_glue.sh
done

# 生成对比表格
python scripts/generate_comparison_table.py \
    eval_encoder/glue_results/glue_results_*.json
```

### 工作流 4: 完整 GLUE 评测
```bash
# 完整 8 任务评测
BACKEND=flashsvd RETENTION=0.5 \
    bash scripts/one_click_glue.sh

# 分析结果
python scripts/analyze_results.py \
    eval_encoder/glue_results/glue_results_*.json \
    --mode details
```

## 📚 相关文档

- [快速开始指南](../docs/user-guides/QUICKSTART.md)
- [后端选择指南](../docs/user-guides/BACKEND_GUIDE.md)
- [保有率参数指南](../docs/user-guides/RETENTION_GUIDE.md)
- [部署指南](../docs/user-guides/DEPLOYMENT.md)
- [用户指南索引](../docs/user-guides/INDEX.md)

## ❓ 常见问题

**Q: 脚本从哪里执行？**
A: 从任何位置执行都可以，脚本会自动检测路径。

**Q: 如何查看脚本帮助？**
A: 大部分 Python 脚本支持 `--help` 参数。

**Q: 脚本执行失败怎么办？**
A: 先运行 `bash scripts/test_setup.sh` 检查环境。

**Q: 结果保存在哪里？**
A: JSON 结果在 `eval_encoder/glue_results/`，日志在 `eval_encoder/logs/`。

---

**最后更新**: 2026-02-13
