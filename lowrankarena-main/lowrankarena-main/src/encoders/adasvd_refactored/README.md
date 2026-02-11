# AdaSVD Refactored Implementation

这是基于 `Refactored-FlashSVD/experiments/BERTAda/` 的重构版AdaSVD实现。

## 文件说明

### 核心文件

- `adaptive_rank_selection.py` - 训练超网络生成 per-operation ranks
- `profile_svd.py` - Naive backend (标准PyTorch实现)
- `profile_flashsvd.py` - FlashSVD backend (Triton优化kernel)
- `adasvd_wrapper.py` - 包装模块，提供简洁的API

### Kernel文件

- `flashsvdattn.py` - FlashSVD attention kernel (Triton)
- `flashsvdffnv1.py` - FlashSVD FFN v1 kernel (Triton)
- `flashsvdffnv2.py` - FlashSVD FFN v2 kernel (Triton)
- `flash_attn_triton.py` - 基础Flash Attention kernel
- `utils_mask.py` - Mask工具函数

## 使用方式

### 1. 通过 run_encoder_benchmark.py

```bash
# AdaSVD + Naive backend
python run_encoder_benchmark.py \
  --method adasvd \
  --budget 0.4 \
  --backend naive \
  --model_id textattack/bert-base-uncased-SST-2

# AdaSVD + FlashSVD backend
python run_encoder_benchmark.py \
  --method adasvd \
  --budget 0.4 \
  --backend flashsvd \
  --model_id textattack/bert-base-uncased-SST-2
```

### 2. 通过 wrapper API

```python
from adasvd_refactored.adasvd_wrapper import (
    train_adasvd_ranks,
    compress_adasvd_naive,
    compress_adasvd_flashsvd
)

# Step 1: 训练生成ranks
ranks_dict = train_adasvd_ranks(
    model=model,
    calib_loader=calib_loader,
    budget=0.4,
    output_dir="ars_out"
)

# Step 2a: Naive backend压缩
model_naive = compress_adasvd_naive(
    model=model,
    ranks_path="ars_out/ranks.json"
)

# Step 2b: FlashSVD backend压缩
model_flash = compress_adasvd_flashsvd(
    model=model,
    ranks_path="ars_out/ranks.json",
    ffn_kernel="v1"
)
```

## 关键改进

### vs 旧版 (src/encoders/BERTAda/)

| 特性 | 旧版 | 新版 (重构) |
|------|------|-------------|
| **实现方式** | MaskedSVDLinear | FWSVDBlock / FlashSVDBlock |
| **FlashSVD支持** | ❌ 不兼容 | ✅ 原生支持 |
| **内存管理** | 🔴 无主动释放 | 🟢 del + empty_cache |
| **参数布局** | 需要mask管理 | 直接低秩分解 |
| **内存效率** | 差 (673 MB) | 好 (预期~300 MB) |
| **速度** | 慢 (53 ms) | 快 (预期~40 ms) |

### 架构对比

**旧版 MaskedSVDLinear:**
```python
# 每个Linear独立替换
class MaskedSVDLinear:
    # 存储 U, s, V + mask
    # Forward需要外部设置 _current_mask
    def forward(x):
        m = self._current_mask  # 必须外部设置！
        ms = m * self.s
        return x @ (U * ms) @ V.t()
```

**新版 FlashSVDBlock:**
```python
# 整层Block替换
class FlashSVDBlock:
    # 分head存储: Pq[H,dm,R], Vq[H,R,dh]
    def forward(x, mask):
        # 使用FlashSVD kernel
        tmp_q = einsum("bmd,hdr->bhmr", x, self.Pq)
        attn_out = flash_svd_attention(tmp_q, ...)

        # 主动释放内存
        del tmp_q, tmp_k, tmp_v
        torch.cuda.empty_cache()

        # FlashSVD FFN kernel
        y = flashsvd_ffn_v1(...)
        return out
```

## 预期性能

基于 DRONE + FlashSVD 的表现，预期 AdaSVD + FlashSVD 应该达到：

| 指标 | 旧版 (MaskedSVDLinear) | 新版 (FlashSVDBlock) | 改善 |
|------|------------------------|----------------------|------|
| **内存** | 673 MB | ~300 MB | **-55%** |
| **速度** | 53.29 ms | ~40 ms | **-25%** |
| **准确率** | 89.06% | ~89% | 持平 |

## 测试

```bash
cd eval_encoder

# 测试wrapper API
python -c "
from adasvd_refactored.adasvd_wrapper import *
print('Import successful!')
"

# 完整基准测试
./test_adasvd_refactored.sh
```

## 输出文件

训练后会生成：

- `ars_out/ranks.json` - Per-operation rank字典
- `ars_out/budget_report.json` - Budget报告
  ```json
  {
    "target_budget": 0.4,
    "achieved_ratio": 0.6383,
    "total_params": 123456789,
    "max_params": 193456789,
    "num_operations": 199
  }
  ```

## 已知问题

1. **Budget控制失效**: 所有budget (0.3-0.7) 都收敛到 63.8%
   - 可能原因：训练步数太少 (400)、学习率、损失权重
   - 需要调整超参数或训练更长时间

2. **支持架构**: 仅支持 BERT 和 RoBERTa
   - ModernBERT 需要额外适配 (RoPE + GeGLU)

## 参考

- 原始论文: AdaSVD (NAACL 2024)
- 基于: Refactored-FlashSVD/experiments/BERTAda/
- 相关: New_src/flashsvd/compression/adasvd.py
