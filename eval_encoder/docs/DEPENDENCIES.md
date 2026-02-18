# 依赖说明

本文档说明 SVD-Benchmark Encoder Evaluation 的所有依赖。

## 📋 完整依赖列表

### Python 依赖

所有 Python 依赖已列在 `requirements.txt` 中：

```
torch==2.0.1           # PyTorch 深度学习框架
triton                 # Triton GPU 编程框架（FlashSVD kernels）
transformers==4.30.0   # HuggingFace Transformers
datasets==2.14.0       # HuggingFace Datasets
evaluate==0.4.0        # 评估指标库
numpy                  # 数值计算
scipy                  # 科学计算
pandas                 # 数据分析
scikit-learn           # 机器学习工具
tqdm                   # 进度条
```

### 系统依赖

- **CUDA 11.8+** - GPU 加速计算
- **NVIDIA GPU** - 至少 8GB 显存
- **Python 3.10+** - Python 解释器
- **Docker + nvidia-docker2** - 容器运行时（Docker 部署）

## 🔍 依赖用途

### 核心依赖

| 包 | 版本 | 用途 | 是否必需 |
|---|------|------|---------|
| torch | 2.0.1 | 模型训练和推理 | ✅ 必需 |
| transformers | 4.30.0 | BERT 模型加载 | ✅ 必需 |
| datasets | 2.14.0 | GLUE 数据集加载 | ✅ 必需 |
| evaluate | 0.4.0 | GLUE 评估指标 | ✅ 必需 |

### 压缩相关

| 包 | 版本 | 用途 | 是否必需 |
|---|------|------|---------|
| numpy | latest | SVD 分解 | ✅ 必需 |
| scipy | latest | 科学计算 | ✅ 必需 |
| scikit-learn | latest | 数据处理 | ✅ 必需 |

### FlashSVD 后端

| 包 | 版本 | 用途 | 是否必需 |
|---|------|------|---------|
| triton | latest | FlashSVD Triton kernels | ⚠️ FlashSVD 后端需要 |

**注意**:
- 使用 `BACKEND=naive` 时，不需要 Triton
- 使用 `BACKEND=flashsvd` 时，必需 Triton + CUDA GPU

### 工具依赖

| 包 | 版本 | 用途 | 是否必需 |
|---|------|------|---------|
| pandas | latest | 结果分析 | ⚠️ 结果分析脚本需要 |
| tqdm | latest | 进度条显示 | ✅ 必需 |

## 📦 内置依赖

### Triton Kernels

已包含在 `kernels/` 目录：

```
kernels/
├── flashsvdattn.py      # FlashSVD Attention kernel
├── flashsvdffnv1.py     # FlashSVD FFN v1 kernel
├── flashsvdffnv2.py     # FlashSVD FFN v2 kernel
└── flash_attn_triton.py # Flash Attention 基础 kernel
```

这些 kernel 文件已内置，**无需额外下载**。

## ✅ 依赖检查

### 自动检查

运行依赖检查脚本：

```bash
python check_dependencies.py
```

输出示例：
```
============================================================
SVD-Benchmark 依赖检查
============================================================

Python 版本:
  3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0]

依赖包检查:
✓ torch                2.0.1
✓ transformers         4.30.0
✓ datasets             2.14.0
✓ evaluate             0.4.0
✓ numpy                1.24.3
✓ pandas               2.0.3
✓ scipy                1.11.1
✓ sklearn              1.3.0
✓ tqdm                 4.65.0
✓ triton               2.0.0

============================================================
✓ 所有依赖已安装!

✓ CUDA 可用: NVIDIA RTX 4090
  CUDA 版本: 11.8
```

### 手动检查

```bash
# 检查 Python 版本
python --version

# 检查 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 检查 Transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 检查 Triton (FlashSVD 需要)
python -c "import triton; print(f'Triton: {triton.__version__}')"
```

## 🐳 Docker 环境

Docker 镜像已包含所有依赖，**无需手动安装**。

**构建说明**: Docker 必须从父目录 (lowrankarena/) 构建以包含所有依赖：

```bash
cd /path/to/SVD-Benchmark/lowrankarena
docker build -f eval_encoder/Dockerfile -t svd-encoder:latest .
```

Dockerfile 会自动：
1. 安装 CUDA 11.8 + cuDNN 8
2. 安装 Python 3.10
3. 安装所有 Python 依赖（从 `requirements.txt`）
4. 复制 eval_encoder/, utils/, src/, kernels/
5. 配置 PYTHONPATH 以支持所有压缩方法

详细说明见 `BUILD_DOCKER.md`。

## 🔧 手动安装

### 标准安装

```bash
# 安装所有依赖
pip install -r requirements.txt
```

### CPU-only 安装（不推荐）

```bash
# PyTorch CPU 版本
pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# 其他依赖
pip install transformers datasets evaluate numpy scipy pandas scikit-learn tqdm
```

**注意**: CPU 模式下 FlashSVD 不可用，只能使用 naive 后端。

### 最小安装

如果只需要基本功能（不包括 FlashSVD）：

```bash
pip install torch transformers datasets evaluate numpy scipy scikit-learn tqdm
```

## ⚠️ 常见问题

### Q1: Triton 安装失败？

**原因**: Triton 需要 CUDA 支持

**解决**:
- 确保安装了 CUDA 11.8+
- 确保有 NVIDIA GPU
- 如果只需 naive 后端，可不安装 Triton

### Q2: torch 版本冲突？

**原因**: 不同 CUDA 版本需要不同 torch 版本

**解决**:
```bash
# CUDA 11.8
pip install torch==2.0.1

# CUDA 12.1
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu121
```

### Q3: transformers 版本太旧/太新？

**推荐**: 使用 4.30.0（已测试）

**兼容**: 4.25.0 - 4.35.0 应该都可以工作

### Q4: CUDA Out of Memory？

**不是依赖问题**，是显存不足：
- 降低 batch size: `BATCH_SIZE=8`
- 使用更小的模型
- 使用 FlashSVD 后端（节省 8% 显存）

## 📊 依赖大小

| 类别 | 大小估计 |
|------|---------|
| PyTorch + CUDA | ~2GB |
| Transformers | ~500MB |
| 其他 Python 包 | ~500MB |
| Triton | ~100MB |
| **总计** | **~3GB** |

## 🔄 依赖更新

### 安全更新

可以更新这些包而不影响功能：
- numpy, pandas, scipy, scikit-learn
- tqdm

### 谨慎更新

这些包更新可能导致兼容性问题：
- torch
- transformers
- datasets
- triton

### 不推荐更新

Docker 镜像已固定版本，建议使用固定版本以确保可重现性。

## 📞 支持

如遇依赖问题：
1. 运行 `python check_dependencies.py` 检查
2. 查看 Docker 日志
3. 检查 CUDA 和 GPU 驱动

---

**依赖已完整！开箱即用！** 🎉
