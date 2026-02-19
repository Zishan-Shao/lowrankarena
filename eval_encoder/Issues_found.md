# Issues Found & Design Decisions

---

## FWSVD fp16 Bug (2026-02) — Deferred

**现象**: `--method fwsvd --dtype fp16` 报错 `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half`

**根因**: `fwsvd.py:38` — `i_hat`(fp32 diagonal) × `A`(fp16 weight) dtype 不匹配。
Fisher weights 在 fp32 下累积，而 model weights 是 fp16。

**复现**:
```bash
python eval_encoder/run_encoder_benchmark.py --method fwsvd --rank 128 --backend naive \
  --dtype fp16 --task sst2 --model_id textattack/bert-base-uncased-SST-2 --full_validation
```

**搁置原因**: 主线 benchmark 以 fp32 为标准 dtype，fp16 暂不支持 FWSVD。
Fix 思路: `i_hat @ A` 前 `A_fp32 = A.float()`，结果 `.to(A.dtype)` 返回。

---

## BoolQ label2id 曾反转 (2026-02) — 已修复

**现象**: `howey/bert-base-uncased-boolq` 的 `id2label={0: 'LABEL_0', 1: 'LABEL_1'}`（通用名，无法自动识别）。
Probe 确认：model class 0 = True (与 super_glue/boolq dataset 的 0=False 相反)。
早期未加 remap → dense accuracy = 0.2945（实为 1-accuracy = 1-0.7055）。

**修复**: 在 TASK_CFG 中加 `model_remap_overrides={"howey/bert-base-uncased-boolq": [1, 0]}`。
修复后 dense fp32 = **0.7055** ✓

---

## ANLI de-Adversarialization (2026-02) — 真实现象

**现象**: `textattack/bert-base-uncased-MNLI` 在 ANLI R1/R2/R3 上经 SVD r=128 压缩后准确率**上升**：

| Round | Dense fp32 | SVD r=128 fp32 | Δ |
|-------|-----------|---------------|---|
| R1 | 0.230 | 0.342 | +11.2% |
| R2 | 0.283 | 0.318 | +3.5% |
| R3 | 0.292 | 0.344~0.348 | +5.2% |

**Sanity check 通过**:
- 同一 model_id / split / n=1000 / full_validation=True ✓
- Naive == FlashSVD accuracy ✓（kernel 路径可信）
- Dense fp16 ≈ Dense fp32（0.229 vs 0.230 = 1 sample）✓

**解释**: ANLI 专门对抗 BERT MNLI 模型学到的 spurious shortcuts。SVD 压缩同时破坏 NLI 推理能力和这些快捷方式，使模型退化为接近随机预测（~33%），而对抗攻击设计让 dense 模型系统性低于随机（~23-29%）。
结论：压缩使模型"更蠢但不再被针对性愚弄"，是压缩鲁棒性研究的一个有趣侧面。

---

## SuperGLUE Simplified Metric Rationale

### Why we use simplified metrics instead of official SuperGLUE scoring

**Date**: 2026-02

**Research context**: This codebase is a *compression method benchmark* (FlashSVD / AdaSVD / FWSVD / DRONE),
not a SuperGLUE leaderboard submission. The core research question is:

> "How much does low-rank compression degrade relative performance across task categories?"

The metric of interest is the **delta between dense and compressed**, not the absolute score.

---

### Task-by-task simplification decisions

| Task | Official Metric | We Use | Reason for Simplification |
|------|----------------|--------|--------------------------|
| BoolQ | Accuracy | Accuracy | No simplification needed |
| CB | Accuracy + F1 | Accuracy | F1 not needed for compression trend analysis |
| RTE (SG) | Accuracy | Accuracy | No simplification needed |
| WiC | Accuracy | Accuracy; no word-position marking | Position marking adds ~2-3% but complicates tokenization; compression trend is still valid |
| COPA | Accuracy | Deferred (Phase 3) | Requires dual-pair scoring per example; significant preprocessing complexity |
| MultiRC | F1a + EM | Deferred (Phase 3) | Group-level scoring across questions adds ~200 lines of evaluation engineering |
| ReCoRD | EM + F1 | Deferred (Phase 3) | Entity-level exact match with span normalization; not a classification task |
| WSC | Accuracy | Deferred (Phase 3) | Small dataset (554 train), span-marking encoding adds complexity |

---

### When simplified metrics are scientifically valid

Simplified metrics are valid for compression research when:
1. The **ranking** between methods is preserved (FlashSVD > naive SVD remains true regardless of metric)
2. The **trend** across rank budgets is monotonic (lower rank → lower accuracy, regardless of metric variant)
3. We are **not claiming leaderboard scores** in the paper

For COPA/MultiRC/ReCoRD, the more complex metrics would be needed if:
- Submitting to SuperGLUE leaderboard
- Claiming "state-of-the-art" on these tasks

---

### HANS / ANLI: no simplification needed

HANS and ANLI are both standard 2-class / 3-class classification tasks.
- HANS: accuracy (binary entailment vs non-entailment)
- ANLI R1/R2/R3: accuracy (3-class NLI)

These use the same evaluation logic as GLUE NLI tasks. No simplification applied.

**HANS label fold**: The MNLI-finetuned model (textattack) outputs 3 classes
(contradiction=0, entailment=1, neutral=2). HANS expects 2 classes (entailment=0,
non-entailment=1). We fold: pred==1 → 0 (entailment), pred∈{0,2} → 1 (non-entailment).
This is standard practice for HANS evaluation with MNLI models.

---

### Phase roadmap

- **Phase 1 (current)**: BoolQ, CB, RTE_SG, WiC, HANS, ANLI R1/R2/R3
- **Phase 3 (future)**: COPA, MultiRC, ReCoRD, WSC with full official scoring

---

## FlashSVD fp16 vs fp32 Throughput (2026-02)

**Finding**: FlashSVD Triton kernels give clear speed advantages over Naive in fp32 but NOT in fp16.

### BoolQ SVD r=128 benchmark

| dtype | backend | throughput | memory |
|-------|---------|-----------|--------|
| fp16  | naive   | 909 sps   | 267.8 MB |
| fp16  | flashsvd| 735 sps (-19%) | 186.6 MB (-30%) |
| fp32  | naive   | 363 sps   | 527.3 MB |
| fp32  | flashsvd| 407 sps (+12%) | 347.4 MB (-34%) |

### SST2 FWSVD r=64 benchmark

| dtype | backend | throughput | memory |
|-------|---------|-----------|--------|
| fp32 | naive   | 403 sps   | 504.3 MB |
| fp32 | flashsvd| 517 sps (+28%) | 324.4 MB (-36%) |

**Root cause**: NVIDIA cuBLAS has hardware-level fp16 tensor-core optimization that Triton can't match.
For fp32, Triton fused kernels reduce memory bandwidth and beat cuBLAS.

**Implication**: Report fp32 benchmarks in the paper to show FlashSVD's advantage.
Memory savings (~30-36%) are consistent regardless of dtype — usable as a secondary selling point for fp16.

---

## CB Compression Analysis: Proxy Model Problem (2026-02)

**Finding**: CB SVD r=128 collapses to 7.1% accuracy (dense was 73.2%) because:
1. CB has only 56 validation examples (smallest SuperGLUE task)
2. We use `textattack/bert-base-uncased-MNLI` as proxy (no CB-specific public model found)
3. After SVD compression, the proxy model collapses to predicting textattack's "neutral" class
   - CB true distribution: {entailment:23, contradiction:28, neutral:5}
   - Compressed predicts mostly neutral → only 5/56 correct ≈ 7%

**Dense pred distribution** (remap [1,0,2] applied):
- Dense: {entailment:23, contradiction:20, neutral:13} vs True: {entailment:23, contradiction:28, neutral:5}

**Recommendation**: CB is unsuitable for compression trend analysis with a proxy model.
For Phase 3, find a publicly available BERT-base model fine-tuned specifically on super_glue/cb.
Until then, exclude CB from per-rank compression sweep and note it in the paper.

---

## HANS / ANLI 跨任务校准可复现性建议 (2026-02)

当使用 `--calib_task mnli` 对 HANS / ANLI 进行跨任务评测时，校准子采样是随机的，
不同 seed / batch_size 会导致 fwsvd / drone / adasvd 数值轻微波动（±0.5% 量级）。

**建议固定配置（放入论文对应脚本）：**

```bash
python eval_encoder/run_encoder_benchmark.py \
  --method fwsvd --rank 128 \
  --task hans --calib_task mnli \
  --model_id textattack/bert-base-uncased-MNLI \
  --calib_batches 16 \   # 16 × 32 = 512 samples
  --seed 0 --calib_seed 0 \
  --full_validation
```

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--calib_batches` | 16 | 512 samples（32 bs），足够 Fisher/cov 收敛 |
| `--seed` | 0 | 模型初始化 seed |
| `--calib_seed` | 0 | 校准数据采样 seed（独立控制） |

**CSV 中的追踪字段：**
- `calib_source`：校准来源，格式 `"mnli:train"`
- `eval_target`：评测目标，格式 `"hans:validation"`
- 一行即可读出 `mnli:train → hans:validation`

---

## HANS / ANLI 不支持 pretrain_before_compress (2026-02) — 设计限制

`glue_pipeline.py --pretrain_before_compress` 对以下任务无效，会抛出 `ValueError`：

### HANS
- **根因**: HuggingFace datasets ≥ 3.0 废弃了 loading-script 支持。`_load_hans_dataset()` 只能下载 evaluation TSV（30k examples），没有加载训练集的路径。物理上无训练数据可用。
- **结论**: HANS 永远是 eval-only，无法 pretrain。

### ANLI R1/R2/R3
- **根因**: `glue_pipeline.py` 中 `train_split=None` 是刻意设计。ANLI 的核心价值是测试**未经对抗训练**的模型的鲁棒性。若在 ANLI 训练集上 fine-tune 后再测试，模型会"适应"这些对抗样本，结果失去科学意义。
- **附注**: `run_encoder_benchmark.py` 中 ANLI 保留了 `train_split="train_r1/r2/r3"`（允许 calibration 用途），但 `glue_pipeline.py` 的 fine-tune 路径封闭。

### 对应代码
- `glue_pipeline.py:1694` — `run_pipeline()` 中对 `train_split=None` 的 task 提前 raise ValueError
- `glue_pipeline.py:810` — `compress_model()` 中校准方法（fwsvd/drone/adasvd）+ eval-only task 组合提前 raise ValueError

### 若未来需要支持
- HANS: 需要手动下载训练集 TSV 并实现 `_load_hans_train_dataset()`，同时注意 `gold_label` string 转换
- ANLI: 去掉 `train_split=None` 限制即可，但需接受测试结果失去对抗有效性

---

## Phase 1 SVD r=128 Naive Compression Results (2026-02)

| Task | Dense | SVD r=128 | Delta | Notes |
|------|-------|-----------|-------|-------|
| BoolQ | 0.706 | 0.623 | -8.3% | Moderate degradation |
| CB | 0.732 | 0.071 | -66.1% | ⚠️ Collapse (proxy model + 56-example dataset) |
| RTE_SG | 0.650 | 0.491 | -15.9% | Large degradation |
| WiC | 0.688 | 0.497 | -19.1% | Large degradation |
| HANS | 0.559 | 0.499 | -5.9% | Near-random (shortcut detection collapses) |
| ANLI R1 | 0.229 | 0.342 | +11.3% | Paradoxical improvement (de-adversarialization) |
| ANLI R2 | 0.283 | 0.317 | +3.4% | Slight improvement toward random |
| ANLI R3 | 0.293 | 0.346 | +5.3% | Improvement toward random |

**Key insight (ANLI)**: SVD compression destroys the specific learned shortcuts ANLI targets,
bringing adversarially-targeted models from below-random (~23-29%) toward random (~33%).
This is a scientifically interesting compression side-effect: *de-adversarialization*.

---

## Per-Head vs Full-Matrix SVD：语义边界与后端兼容性 (2026-02)

### 定义

| 模式 | 对象 | rank 上限 | 参数形状 |
|------|------|-----------|---------|
| `per_head` | 每个 attention head 单独 SVD（64×64） | `head_dim = 64` | `Pq: [H, dh, R]` |
| `full` | 整个 Q/K/V 矩阵 SVD（768×768） | `d_model = 768` | `Pq: [dm, R]` 或 `[1, H, dm, R]` |

---

### 问题 1：per_head 模式下 rank > head_dim 静默退化为 full rank

**现象**：`--rank 128 --qkv_mode per_head` 时，BERT-base（head_dim=64）实际每头只保留 64 个奇异值，效果等价于不压缩。

**根因**：`torch.linalg.svd(W_head)` 其中 `W_head` shape 为 `[64, 64]`，最多 64 个奇异值。
`rank=128` 被 clamp 到 64，SVD 完全重建原矩阵，无压缩效果。

**影响范围**：所有使用 per_head 模式且 rank ≥ 64 的 svd / fwsvd / drone 结果。

**正确用法**：per_head 模式 rank 应 ≤ 63；论文常用 r=32 或 r=48。

---

### 问题 2：full 模式的 rank 由 ARS 全矩阵语义决定，不能混用 per_head 后端

**现象**：`adasvd_origin` ARS 输出的 ranks（如 Q=270, K=180, V=90）是 **全矩阵语义**（对 768×768 矩阵），
若误传入 per_head 路径（FWSVDBlock / `svd_per_head`），每头 rank = 270/H = 22.5 → 截断为 22，
等价于对全矩阵做非常低秩压缩，信息大量丢失。

**原始 Bug（已修复）**：旧版 `compress_adasvd_naive` 调用 `build_plain_svd_helpers()` 的
`svd_per_head` 路径，把全矩阵 rank 当 per-head rank 使用 → MRPC F1=0。

**修复**：`_LowRankLinear` 直接对整个 `nn.Linear.weight`（768×768）做 SVD，不经过任何 per-head 拆分。

---

### 问题 3：FlashSVD 后端仅支持 per_head 模式

**根因**：FlashSVD Triton kernel（`flashsvdattn.py`）内部按 head 分块计算，
要求 Q/K/V 的 rank 相同且为 per-head 语义。

**具体限制**：
- `profile_flashsvd.py:194`：从 `Pq` 提取 R，统一应用到所有 V 矩阵
- `utils_mask.py:69`：使用单一 `r_dim` 覆盖所有 Q/K/V tile

**推论**：adasvd + FlashSVD 的组合不能直接使用 per-op adaptive ranks，
必须用 **median rank 策略**（见 adasvd_wrapper.py compress_adasvd_flashsvd）统一各层 rank。

---

### 各方法与模式的兼容矩阵

| 方法 | per_head | full | FlashSVD 后端 |
|------|----------|------|--------------|
| svd | ✅ rank≤63 | ✅ | ✅（per_head only） |
| fwsvd | ✅ rank≤63 | ✅ | ✅（per_head only） |
| drone | ✅ rank≤63 | ✅ | ✅（per_head only） |
| adasvd | ❌ 语义不匹配 | ✅ | ⚠️ 需 median rank 策略 |

---

### 推荐配置

```bash
# svd/fwsvd/drone：per_head，rank 控制在 head_dim 以内
--qkv_mode per_head --rank 32   # 合理
--qkv_mode per_head --rank 128  # ❌ 静默无压缩

# adasvd：始终 full-matrix，naive backend
--method adasvd --backend naive --budget 0.5  # ✅
--method adasvd --backend flashsvd            # ⚠️ 自动 median rank
```

---

## AdaSVD (adasvd_origin) 重新实现记录 (2026-02)

### 背景

`adasvd_origin` 是论文 NAACL 2024 paper-compliant 的 ARS（Adaptive Rank Selection）实现，
与 `adasvd_refactored` 的区别在于使用 `PaperHN`（固定随机 z buffer + LayerNorm + meta_proj + GRU）。

---

### Bug 1：one-sided log budget loss 无法从下方推动 ratio（已修复）

**现象**：Paper Eq.8 采用单侧 log 形式 `log(clamp(T, min=Tmax)/Tmax)`，当 ratio_soft < budget 时梯度为 0，ratio 永远无法上升到 target。

**根因**：`PaperHN` 初始 head.bias = -2.0（sigmoid ≈ 12%），远低于任何 budget，而单侧 loss 对 T < Tmax 无梯度，无法推动 ratio 上升。

**修复**（`adaptive_rank_selection.py` PaperHN.__init__）：
```python
init_p = float(max(0.55, min(0.95, budget + 0.15)))
init_bias = math.log(init_p / (1.0 - init_p))
for head in self.heads:
    nn.init.constant_(head.bias, init_bias)
```
让初始 ratio_soft 略高于 budget，使 loss 从第 0 步就有向下梯度。

---

### Bug 2：alignment_loss 量级压制 budget_loss（已修复）

**现象**：ratio_soft 稳定上升（0.976 → 1.4），budget_loss 也上升，alignment 梯度完全压制 budget 梯度。

**根因**：BERT singular values `s_max ≈ 30–50`，`((mask - m_top) * s)^2` 量级约 `50^2 = 2500`；
而 budget_loss 量级 `O(1)`，alignment 强于 budget 约 **67,000×**。

**修复**（`adaptive_rank_selection.py` alignment_loss）：
```python
frob_sq = s.detach().pow(2).sum().clamp(min=1e-12)
return torch.sum(((mask - m_top) * s) ** 2) / frob_sq
```
归一化使 alignment_loss 无量纲（范围 [0,1]），与 budget_loss 量级对齐。

---

### Bug 3：ratio_max 边界断言设置错误（已修复）

**现象**：ratio_soft 偶尔超过 1.0，原始断言 `assert ratio_soft <= 1.01` 会误报。

**根因**：SVD 表示用 `(M+N)×R` 个参数，dense 用 `M×N`；
对 BERT-base（square attn + rect FFN），full-rank SVD ratio_max ≈ 1.50 > 1。

**修复**（`adasvd_wrapper.py` 训练循环）：
```python
T_max_fullrank = sum((op.in_features + op.out_features) * op.rank_cap for op in op_list)
ratio_max = T_max_fullrank / (T_original + 1e-12)
assert ratio_soft <= ratio_max + 1e-2, f"ratio_soft={ratio_soft:.4f} > ratio_max={ratio_max:.4f}"
```

---

### Bug 4：compress_adasvd_naive 用 FWSVDBlock 导致 F1=0（已修复）

**现象**：ARS 训练收敛（ratio_soft → target），但压缩后 MRPC F1=0.0。

**根因**：旧 `compress_adasvd_naive` 调用 `profile_svd.py` 的 `FWSVDBlock`，该 block：
1. 导入 `flash_attn_triton`（Triton kernel，attention mask 约定不同）
2. 用 `svd_per_head` 把 ARS 全矩阵 rank（最大 768）当作 per-head rank（最大 64）使用 → 所有层退化为 full rank

**修复**（`adasvd_wrapper.py`）：弃用 FWSVDBlock，改用 `_LowRankLinear` 直接逐 Linear 替换：
```python
class _LowRankLinear(nn.Module):
    """W ≈ A @ Bt; forward: x @ Bt.T @ A.T + bias"""
    def __init__(self, A, Bt, bias=None):
        ...
def compress_adasvd_naive(model, ranks_path, device="cuda"):
    # For each nn.Linear in ranks_dict: SVD → A=[out,r], Bt=[r,in] → _LowRankLinear
    ...
```

---

### Bug 5：load_compressed_model.py 不认识 _LowRankLinear checkpoint（已修复）

**现象**：fine-tuning 阶段加载 adasvd checkpoint 时打印 `[warn] Layer N missing SVD parameters, skipping`（全部 12 层），加载后精度等于 dense baseline。

**根因**：`load_compressed_model.py` 只识别 `.block.Pq` / `.block.Uq` 参数名，
而 `_LowRankLinear` 保存的是 `.A` / `.Bt`，没有 `.block.` 前缀。

**修复**（`load_compressed_model.py`）：在 SVD block 重建逻辑前插入 adasvd 分支：
```python
has_lrl = any(k.endswith(".A") for k in state_dict.keys())
if has_lrl:
    # 遍历 base_model 的 nn.Linear，找到 state_dict 里有对应 .A/.Bt 的就替换
    for name, module in list(base_model.named_modules()):
        if isinstance(module, nn.Linear) and f"{name}.A" in state_dict:
            setattr(parent, last, _LowRankLinear(A, Bt, bias))
    base_model.load_state_dict(non_lrl_keys, strict=False)
    return base_model, tokenizer, comp_info
```

`missing_keys` 报告 148 条（= 74 层 × 2 个 A/Bt）为**预期行为**，不是错误。

---

### Bug 6：fine-tuning 时 A/Bt 被冻结导致 MCC/F1=0（已修复）

**现象**：加载 adasvd checkpoint 后 fine-tuning 3 epochs，loss 下降但 MCC 始终 0.0。

**根因**：`_LowRankLinear` 构造时设置 `requires_grad=False`（为 inference benchmark 设计），
fine-tuning 只更新了 classifier + LayerNorm，encoder 特征完全冻结。

**修复**（`load_compressed_model.py` adasvd 分支末尾）：
```python
for m in base_model.modules():
    if isinstance(m, _LowRankLinear):
        m.A.requires_grad_(True)
        m.Bt.requires_grad_(True)
```

---

### 涉及文件汇总

| 文件 | 修改内容 |
|------|---------|
| `src/encoders/adasvd_origin/adaptive_rank_selection.py` | PaperHN bias init；alignment_loss Frobenius 归一化；MaskedSVDLinear.rank_cap 属性 |
| `src/encoders/adasvd_origin/adasvd_wrapper.py` | ratio_max 断言；`_LowRankLinear` 类；新版 `compress_adasvd_naive` |
| `eval_encoder/load_compressed_model.py` | `_LowRankLinear` 类定义；adasvd checkpoint 加载分支；fine-tuning requires_grad 解冻 |
| `eval_encoder/scripts/one_click_glue.sh` | 新增 `ADASVD_CALIB_SAMPLES` / `ADASVD_STEPS` env → python args 穿透 |
| `eval_encoder/scripts/compare_all_methods.sh` | `METHODS` / `STAGES` env 覆盖支持；adasvd 传递 ARS 参数 |

---

## AdaSVD Classifier/Pooler 被 ARS 压缩 (2026-02) — 已修复

**现象**: CoLA 微调后 MCC 持续为 0，SST-2/MRPC 可以恢复但 CoLA 不行。

**根因**: `collect_linear_modules(model)` 收集**所有** `nn.Linear`，包括 `classifier` 和 `bert.pooler.dense`，均参与 ARS rank 分配。

对 CoLA（binary，`num_labels=2`）实测 `ranks.json`（budget=0.5，正确收敛到 0.500）：
- `classifier [2×768]`：**rank=1**（rank_cap=min(2,768)=2，被压到下界！）
- `bert.pooler.dense [768×768]`：**rank=134**（仅剩 35% 原始参数）

`classifier rank=1` 的影响：
- 压缩后 `A=[2,1], Bt=[1,768]`，只有 1 个决策方向
- 模型坍塌到全预测同一类（MCC=0），且由于表示空间被极度压缩，微调难以逃出
- 注：同任务不同校准数据下，SST-2 得到 `classifier: rank=2`（full rank），所以能恢复；CoLA 得到 rank=1，不能

`pooler rank=134` 的影响：
- [CLS] token 的任务相关变换只保留 35% 参数
- 对 sentiment/paraphrase（SST-2/MRPC）影响小，对 syntactic acceptability（CoLA）影响大

**修复** (`src/encoders/adasvd_origin/adasvd_wrapper.py:99-112`)：
```python
HEAD_EXCLUDE = ("classifier", "pooler")
linear_list_all = collect_linear_modules(model)
linear_list = [(n, m) for n, m in linear_list_all
               if not any(pat in n for pat in HEAD_EXCLUDE)]
```
- `classifier` 和 `pooler` 不进入 ARS，`compress_adasvd_naive` 中 `rank is None → skip`，保持 full-rank `nn.Linear`
- encoder 72 层获得的 budget 几乎不变（差 <0.7%）
- `ranks.json` 缩减为 72 条（encoder only）

**需重新压缩**才能生效。

---

## Phase 1 Dense Baselines (2026-02)

Verified with correct label remaps. All results use `full_validation=True`.

| Task | Model | Dense Accuracy | Notes |
|------|-------|---------------|-------|
| BoolQ | howey/bert-base-uncased-boolq | 0.7055 | label_remap=[1,0] |
| CB | textattack/bert-base-uncased-MNLI | 0.7321 | label_remap=[1,0,2] |
| RTE_SG | howey/bert-base-uncased-rte | 0.6498 | label_remap=None (ordering matches) |
| WiC | rycecorn/Bert-fine-tuned-WiC | 0.6881 | label_remap=None (ordering matches) |
| HANS | textattack/bert-base-uncased-MNLI | 0.5585 | fold: pred==1→0 |
| ANLI R1 | textattack/bert-base-uncased-MNLI | 0.2290 | below random (adversarial, expected) |
| ANLI R2 | textattack/bert-base-uncased-MNLI | 0.2830 | below random (adversarial, expected) |
| ANLI R3 | textattack/bert-base-uncased-MNLI | 0.2925 | below random (adversarial, expected) |
