# ModernBERT Integration Plan
# Last updated: 2026-03-02

## 背景

eval_encoder/ 是主 benchmark pipeline。ModernBERT 已有 SVD + naive backend，
其余方法（FWSVD/DRONE/AdaSVD）和 FlashSVD backend 全部被硬编码 raise error 拦截。

---

## 现状矩阵

| 方法        | BERT/RoBERTa | ModernBERT | 拦截位置 |
|-------------|:---:|:---:|---|
| SVD         | ✅  | ✅  | — |
| FWSVD       | ✅  | ❌  | `run_encoder_benchmark.py` `_build_fwsvd()` ~line 462 |
| DRONE       | ✅  | ❌  | `run_encoder_benchmark.py` `_build_drone()` ~line 608 |
| AdaSVD      | ✅  | ❌  | `run_encoder_benchmark.py` `_build_adasvd()` ~line 647 |
| naive       | ✅  | ✅  | — |
| sdpa        | ✅  | N/A | ModernBERT block 本身就用 SDPA |
| flashsvd    | ✅  | ❌  | `flashsvd_backend.py` `enable_flashsvd()` ~line 231 |
| flashsvd15  | ✅  | ❌  | `flashsvd_backend.py` `enable_flashsvd15()` ~line 413 |

---

## 关键架构差异（决定所有 porting 工作）

| 方面 | BERT/RoBERTa | ModernBERT |
|------|---|---|
| 层路径 | `model.bert.encoder.layer[i]` | `model.model.layers[i]` |
| Norm 位置 | post-norm（残差后） | pre-norm（操作前） |
| QKV | 三个独立 Linear `[D,D]` | fused `Wqkv [3D, D]` |
| 位置编码 | 绝对位置 embedding | RoPE (`layer.attn.rotary_emb`) |
| FFN 激活 | GELU | GeGLU (`Wi` 输出 `[2D]`，split gate+linear) |
| attn_norm | 无（post-norm 模型没有） | `layer.attn_norm` |
| mlp_norm | 无 | `layer.mlp_norm` |
| attention out | `layer.attention.output.dense` | `layer.attn.Wo` |
| FFN in | `layer.intermediate.dense` | `layer.mlp.Wi` (out=[2D]) |
| FFN out | `layer.output.dense` | `layer.mlp.Wo` |

---

## Phase 1 — FWSVD for ModernBERT

### 改动文件
- `utils/encoder_utils/fwsvd.py` — 新增 2 个函数
- `eval_encoder/run_encoder_benchmark.py` — `_build_fwsvd()` 去掉 error 加分支

### Step 1.1：`utils/encoder_utils/fwsvd.py` 新增

```python
def estimate_fisher_weights_modernbert(model, dataloader, device):
    """
    与 estimate_fisher_weights_bert_with_attention() 对称。

    ModernBERT 层子模块：
      layer.attn.Wqkv  — fused [3D, D]，split 成 Q/K/V 再算 Fisher
      layer.attn.Wo    — [D, D]
      layer.mlp.Wi     — [2D, D]（GeGLU，出口是 gate+linear 拼接）
      layer.mlp.Wo     — [D, D]

    返回：fisher_wqkv, fisher_wo_attn, fisher_wi, fisher_wo_ffn
    每个都是 {layer_idx: Tensor} 的 dict，值已归一化到 [0,1]。

    注意：Wqkv.grad 的 shape 是 [3D, D]，
    split 成三份 [D, D] 分别平方求和再平均，得到 fisher_wqkv[i]。
    """
    ...

def build_fwsvd_helpers_modernbert(model, dataloader, device, eps=1e-6):
    """
    返回 (per_head_fn, low_rank_fn)，与 build_fwsvd_helpers() 接口一致。

    内部：
    1. 调用 estimate_fisher_weights_modernbert() 得到 4 个 fisher dict
    2. 建立 data_ptr() → fisher_vector 映射 fw_map
       - Wqkv：先 split，分别注册 Q/K/V 三个 ptr 指向同一 fisher_wqkv[i]
         (因为 NaiveModernBertSVDBlock 对 Q/K/V 分别调用 per_head_fn)
       - Wi：split 出 gate/linear 两半，各自注册到 fw_map
         (NaiveModernBertSVDBlock 对整个 Wi 调用 low_rank_fn)
         → 统一用 Wi 整体的 fisher 即可，不 split
    3. 闭包内 compute_row_sum_svd_decomposition() 与 BERT 版本完全相同
    """
    ...
```

### Step 1.2：`run_encoder_benchmark.py` `_build_fwsvd()`

```python
# 将:
if arch == "modernbert":
    raise RuntimeError("FWSVD is not yet supported for ModernBERT...")

# 改为:
if arch == "modernbert":
    from utils.encoder_utils.fwsvd import build_fwsvd_helpers_modernbert
    return build_fwsvd_helpers_modernbert(model, calib_loader, device)
```

---

## Phase 2 — DRONE for ModernBERT

### 改动文件
- `eval_encoder/run_encoder_benchmark.py` — 新增 `_calibrate_covariances_modernbert()` + `_build_drone()` 加分支

### Hook 点对照

| 数据 | BERT hook 位置 | ModernBERT hook 位置 |
|------|---|---|
| attn_in | `layer.attention.self` 输入 | `layer.attn_norm` **输出** |
| attn_out | `layer.attention.output.dense` 输入 | `layer.attn.Wo` 输入 |
| ffn_in | `layer.intermediate.dense` 输入 | `layer.mlp_norm` **输出** |
| ffn_out | `layer.output.dense` 输入 | `layer.mlp.Wo` 输入 |

**GeGLU 特殊处理**：
`layer.mlp.Wi` 输出 shape 是 `[B, M, 2D]`（gate + linear concat）。
做 covariance 时只用前半 `[:D]`（linear 路，真正流向下游的激活）。

### Step 2.1：新增函数

```python
def _calibrate_covariances_modernbert(model, loader, device, encoder_layers, max_batches=4):
    """
    与 _calibrate_covariances() 接口相同，返回相同结构的 layer_covs dict。

    差异：
    - hook 挂在 attn_norm.output, attn.Wo.input, mlp_norm.output, mlp.Wo.input
    - ffn_in covariance 用 mlp_norm 输出（不是 Wi 输出）
    - 不需要处理 GeGLU split（Wi 输入是正常激活，输出才是 [2D]）
    """
    ...
```

### Step 2.2：`_build_drone()` 加分支

```python
if arch == "modernbert":
    layer_covs = _calibrate_covariances_modernbert(model, calib_loader, device, encoder_layers)
else:
    layer_covs = _calibrate_covariances(model, calib_loader, device, encoder_layers)
# 后续 _data_aware_low_rank() 调用不变（接口一致）
```

---

## Phase 3 — AdaSVD for ModernBERT

### 改动文件
- `src/encoders/BERTAda/adaptive_rank_selection.py` — `SimpleHN` 支持 `n_ops` 参数
- `eval_encoder/run_encoder_benchmark.py` — `_build_adasvd()` 去掉 error 加分支

### 关键差异

| 项目 | BERT (6 ops/layer) | ModernBERT (4 ops/layer) |
|---|---|---|
| ops | Q, K, V, Wo, Wi, Wo_ffn | Wqkv, Wo, Wi, Wo_ffn |
| budget Tmax | `sum((M+N)*R)` 修复后 `sum(M*N)` | `sum(3D² + D² + 2D² + D²) = 7D²` per layer |
| rank 输出 | 6个rank per layer | 4个rank，Wqkv rank → 拆成 Q/K/V 同一 rank |

### Step 3.1：`adaptive_rank_selection.py`

```python
class SimpleHN(nn.Module):
    def __init__(self, n_layers, n_ops=6, feat_dim=64, hidden=128):
        # n_ops: BERT=6, ModernBERT=4
        # 修改 heads 的数量从硬编码 6 → n_ops
        ...
```

Budget 计算改为接收 `op_sizes` 列表而非硬编码维度：
```python
# 调用时传入每个 op 的 M*N：
# ModernBERT: [3D², D², 2D², D²]
Tmax = p * sum(op_sizes)
```

### Step 3.2：`run_encoder_benchmark.py` `_build_adasvd()`

```python
if arch == "modernbert":
    # n_ops=4: [Wqkv, Wo, Wi, Wo_ffn]
    ranks = train_adasvd_ranks(model, calib_loader, budget,
                               arch="modernbert", n_ops=4, ...)
    # ranks 输出: {layer: [r_wqkv, r_wo, r_wi, r_wo_ffn]}
    # 建块时: r_wqkv 用于 Q/K/V 三个 per_head_fn（rank 相同）
```

---

## Phase 4 — Benchmark 集成

### Step 4.1：确认 ModernBERT task model IDs

需要搜索 HuggingFace 是否有 fine-tuned 版本，例如：
- `answerdotai/ModernBERT-base`（base，无 task fine-tuning）
- 第三方 fine-tuned（搜索 `ModernBERT MNLI` 等）

若无现成模型，需先跑 fine-tuning（或用 base model zero-shot）。

在 `run_encoder_benchmark.py` 的 `_TASK_MODELS` 补充：
```python
_TASK_MODELS_MODERNBERT = {
    "mnli": "???",
    "stsb": "???",
    "cola": "???",
    ...
}
```

### Step 4.2：运行实验

```bash
# 先验证 Phase 1 (FWSVD)
python eval_encoder/run_encoder_benchmark.py \
    --method fwsvd --rank_attn 48 --rank_ffn 256 --rank_wo 208 \
    --task mnli --model_id <modernbert_mnli> \
    --backend naive --dtype bf16 --seq_len 512

# 全量 GLUE（Phase 1+2 完成后）
METHODS="svd fwsvd drone" TASKS="mnli stsb cola sst2 mrpc qqp qnli rte" \
MODEL_ARCH=modernbert bash eval_encoder/scripts/compare_all_methods.sh
```

### Step 4.3：图表

- 在现有 `plot_combined_figure.py` 加 ModernBERT 列，或
- 单独出一张 ModernBERT vs BERT-base 对比图

---

## Phase 5 — FlashSVD for ModernBERT（可选）

需要新建 Triton kernels：
- `kernels/encoder_kernels/flashsvdropeattn.py`：rank-space attention + RoPE fused
- `kernels/encoder_kernels/flashsvdgeglu.py`：GeGLU FFN fused

工程量极大，且 ModernBERT 本身已用 SDPA（加速空间相对小）。
**建议论文里标 "future work"，当前优先 Phase 1-4。**

---

## 执行顺序

```
Phase 1 (FWSVD)   ← 立即可做，约 80-100 行新代码，1 天
Phase 2 (DRONE)   ← 与 Phase 1 并行，换 hook 点，1 天
Phase 4.1         ← 同时确认 task model IDs（不需要写代码）
Phase 3 (AdaSVD)  ← Phase 1+2 之后，改 hypernetwork，2-3 天
Phase 4.2-4.3     ← 跑实验出图
Phase 5           ← 可选
```

---

## 相关文件索引

| 文件 | 用途 |
|------|------|
| `eval_encoder/blocks.py` | `NaiveSVDBlock`, `NaiveModernBertSVDBlock` 定义 |
| `eval_encoder/flashsvd_backend.py` | FlashSVD backend 启用，有 ModernBERT error |
| `eval_encoder/run_encoder_benchmark.py` | 主 benchmark 入口，压缩方法构建，有 3 处 error |
| `eval_encoder/glue_pipeline.py` | 多任务 fine-tune + 评估 pipeline |
| `eval_encoder/load_compressed_model.py` | checkpoint 加载，已支持 ModernBERT |
| `utils/encoder_utils/fwsvd.py` | FWSVD Fisher 计算，需新增 modernbert 版本 |
| `utils/encoder_utils/svd_helpers.py` | `build_fwsvd_helpers()` 入口 |
| `src/encoders/BERTAda/adaptive_rank_selection.py` | AdaSVD hypernetwork，需加 n_ops 参数 |
| `src/encoders/ModernBERT/BERT_MASK/run_modernbert_svd.py` | ModernBERT 层结构参考 |
| `eval_encoder/scripts/analyze_compute.py` | FLOPs 分析，已支持 naive+sdpa，不支持 ModernBERT 的 Pq 结构分析（单独问题） |
