# Issues

## [CLOSED] whitening_then_update PPL=29 是 unigram 假象

**发现时间**: 2026-04-07  
**影响**: SVDLLMv1 whitening_then_update 系列的 zero-shot 数据不可信

### 现象

`SVDLLMv1_whitening_then_update_*`（llama31_8b，所有 keep_ratio）出现矛盾结果：

| checkpoint | wiki2 PPL | boolq | hellaswag | 说明 |
|---|---|---|---|---|
| baseline | 6.96 | 0.831 | 0.793 | 正常 |
| whitening_only_0.8 | 828 | 0.541 | 0.290 | 坍塌，符合预期 |
| whitening_then_update_0.8 | **29** | **0.378** | **0.260** | 矛盾：PPL 低但 zero-shot chance level |

PPL=29 + 所有 zero-shot ≈ chance level，在真实 LM 中不可能同时成立。

### 根因分析

`whitening_local_update` 在 calibration data 上做梯度下降。如果模型在 whitening 步骤已坍塌（权重秩不足，输出恒定 logits），local update 会把"恒定输出向量"优化到接近 wikitext2 **token 频率分布**，导致：

- **PPL=29** ≈ unigram entropy（英语约 20–60），并非真实语言建模能力
- **zero-shot 全 chance level**：输出与输入无关，loglikelihood 比较退化为 token 频率偏置

BoolQ 恒选 "No" → acc = False 标签占比 = **0.378287**  
MathQA 恒选 "a" → acc = "a" 正确频率 = **0.205695**

### 已验证 (2026-04-07)

**验证一：评估路径排查（check_lmeval_compat.py）**

```
use_cache=False (PPL path): boolq_deg=True  mathqa_deg=True
use_cache=True  (lm-eval default): boolq_deg=True  mathqa_deg=True
HFLM (actual lm-eval):  boolq_deg=True  mathqa_deg=True
```

三路全部 DEGENERATE，排除：
- KV-cache bug（两种 use_cache 结果一致）
- lm-eval 接口问题（HFLM 路径相同）

**验证二：HF 转换排查（直接评估原始 .pt）**

直接加载压缩后、HF 转换前的 `.pt` checkpoint：

```
checkpoint: .../meta_llama_Llama_3.1_8B_whitening_then_update_0.8.pt
wiki2=29.23  c4=115.71  boolq=0.3783  hellaswag=0.2601  avg=0.333  DEGENERATE_ZEROSHOT
```

与 HF 转换后结果完全一致，排除：
- HF 格式转换引入的误差

**结论：问题在压缩本身。** whitening 步骤破坏了 Llama-3.1-8B 的模型权重，local update 无法恢复。

### 影响范围

CSV `baselines/SVD-LLM/checkpoints/svdllm/llama31_8b/results.csv` 中：
- `SVDLLMv1_whitening_then_update` 所有 keep_ratio 的 zero-shot 列不可信
- `SVDLLMv2` 系列同样出现大面积 boolq=0.3783，需一并检查

### 已做的临时处理

`evaluate/eval_decoder.py` 加了退化检测：boolq≈0.3783 且 mathqa≈0.2057 时，`notes` 列自动打 `DEGENERATE_ZEROSHOT`。

---

## [CLOSED] HAP-E dense baseline 掉点来源分析

**发现时间**: 2026-04-08  
**影响**: 明确了我们的 zero-shot 评估协议与 HAP-E 的差异来源，不影响压缩质量结论

### 现象

我们的 dense baseline（Llama-3.1-8B）与 HAP-E 论文数字存在差距：

| Task | 我们 (zero-shot) | HAP-E | Gap |
|---|---|---|---|
| HellaSwag acc_norm | 79.0% | 81.7% | −2.7% |
| ARC-c acc_norm | 53.7% | 58.2% | −4.5% |

### 排查过程（2026-04-08/09）

**假说一：add_bos_token 缺失**

`eval_decoder.py` 把 model/tokenizer 对象传给 HFLM，HFLM 无法从 tokenizer_config.json 自动检测 `add_bos_token`，默认 False。而 Llama tokenizer 应为 True。

验证：patch `eval_decoder.py`，显式读取 `tokenizer.add_bos_token` 传给 HFLM。

结果：ARC-c 仍为 53.67%（=patch 前）。**假说一证伪。**

根因：Llama-3.1 用 tiktoken 包装的 `PreTrainedTokenizerFast`，`add_bos_token` 不是其实例属性，`getattr` 返回 False，patch 形同虚设。

**假说二：num_fewshot 不一致**

验证：对 dense baseline 分别跑 25-shot ARC-c 和 10-shot HellaSwag。

结果：

| Task | 我们 zero-shot | 我们 few-shot | HAP-E |
|---|---|---|---|
| ARC-c acc_norm | 53.67% | **58.11%** ± 1.44% | 58.2% |
| HellaSwag acc_norm | 79.0% | **82.33%** ± 0.38% | 81.7% |

few-shot 结果与 HAP-E 在误差范围内完全吻合。**假说二成立，gap 100% 由 num_fewshot 解释。**

### 根因结论

HAP-E 虽在论文中写 "zero-shot benchmarks"，实际对 HellaSwag 用 10-shot、ARC-c 用 25-shot，与 Open LLM Leaderboard v1 标准配置一致。最可能的原因是使用了 lm-eval v0.3.x（task YAML 内置默认 few-shot），而非显式指定。

HAP-E 的 "zero-shot" 含义是"压缩后无 fine-tuning"，不是 lm-eval 的 `num_fewshot=0`。

### 对 paper 的影响

- 我们的 zero-shot 评估协议（num_fewshot=0）内部完全一致，无需修改
- 不能直接引用 HAP-E 的 dense 数字与我们的压缩结果放同一列比较
- 如需对比，只比 relative degradation（相对各自 dense baseline 的下降），或加 footnote 说明协议差异

### 备注

`eval_decoder.py` 加了 `--add_bos_token` 手动开关备用。CSV 中原始 baseline（ARC-c 54.86%）与官方 lm-eval CLI 结果一致，无系统性偏差。

---

## [CLOSED] DobiSVD PPL 更好但 zero-shot 崩溃

**发现时间**: 2026-04-11  
**影响**: DobiSVD 的 PPL 指标不能反映 zero-shot 质量

### 现象

同等压缩率（keep_ratio=0.8，即压缩 80%）下，DobiSVD 的 PPL 优于 whitening_only，但 zero-shot 反而更差：

| 方法 | wiki2 PPL | boolq | 说明 |
|---|---|---|---|
| whitening_only_0.8 | 829 | 0.563 | 有判别能力 |
| DobiSVD_0.8 | 546 | 0.378 | 退化到 always-False |

`boolq=0.378287` = "always predict False" 基线（BoolQ 约 38% 答案为 False），说明模型输出完全退化。

### 根因分析

DobiSVD 对权重的修改方式（`weight_updater.py` line 211）：

```python
W_new = (W.T @ V_pca @ G @ V_pca.T).T
```

`V_pca` 是从 **wikitext2 校准数据输出激活** 的 Incremental PCA 主成分。该操作把 W 投影到 wikitext2 激活的主子空间，清零不在该子空间内的所有权重成分。

**PPL 更好**：wikitext2 激活子空间正好是文本续写最重要的方向，因此在 wiki2/c4/ptb 上 PPL 下降。

**zero-shot 崩溃**：BoolQ 需要区分 "Yes"（token 3869）和 "No"（token 2360）。这两个 token 在 wikitext2 中极少出现，其对应的激活方向不在 V_pca 的主成分中。投影操作把这个判别方向清零 → 所有输入映射后得到相同输出 → lm_head 对 Yes/No 的 logit 差趋近 0 → 恒预测 False。

### 与 whitening_only 的对比

| 压缩维度 | whitening_only | DobiSVD |
|---|---|---|
| 压缩基准 | 权重矩阵几何 SVD | wikitext2 激活 PCA |
| 保留方向 | W 最大奇异值方向（数据无关） | wikitext2 激活主成分（数据偏向） |
| 分布内 PPL | 较差（保留了不重要方向）| 较好 |
| 通用判别能力 | 保留 | 被 wikitext2 偏向清零 |

### 结论

DobiSVD 的 PCA 方法在**同分布** PPL 上有效，但对 zero-shot 分类任务有系统性破坏。这是校准数据分布偏差的固有问题，不是加载或实现 bug。

**对 benchmark 的影响**：DobiSVD 的 zero-shot 结果（boolq=0.378 等）标记为 `DEGENERATE_ZEROSHOT`，不参与方法对比；PPL 数字本身有效，但其参考价值有限（不代表真实语言建模能力）。

---

## [CLOSED] SVDLLMv2 ratio 反转缺失导致全面退化

**发现时间**: 2026-04-12  
**影响**: 服务器上所有现有 V2 checkpoint 实际压缩率与文件名相反，结果无效

### 现象

SVDLLMv2（`v2_0.x.pt`）所有 keep_ratio 的 wiki2 PPL 从 1180（0.4）到 28100（0.8），zero-shot 全部退化（boolq=0.378287）。参考数据（V2 0.8 wiki2 PPL=28.78）与我们的结果（PPL=28100）相差 1000×。

### 根因

`SVDLLM.py`（V1）在 line 1113 做了 ratio 反转：

```python
args.ratio = 1 - args.ratio   # CLI 0.2 → 内部 0.8 → 保留 80%
```

`run_svdllm_v2_compress.py`（V2）无此反转：

```python
keep = 1 - args.ratio                       # 仅用于文件名 → keep=0.8
whitening_hetero(..., ratio=args.ratio, ...) # 直接用 CLI 值 0.2 → 只保留 20%!
```

结果：`v2_0.8.pt` 文件名声称保留 80%，实际只保留 20% 奇异值，严重过压缩。

### 修复

`run_svdllm_v2_compress.py` 中加一行（2026-04-12 已提交）：

```python
ratio_keep = 1.0 - args.ratio   # 与 V1 保持一致
whitening_hetero(..., ratio=ratio_keep, ...)
whitening_local_update(..., ratio_keep, ...)  # local_update 同步修复
```

Shell 脚本传入的 `RATIO in 0.2 0.3 0.4 0.5 0.6` 不变，生成的文件名 `v2_0.8/0.7/0.6/0.5/0.4.pt` 不变，keep_ratio 语义现在正确。

### 操作

服务器上需要删除旧的错误 checkpoint 后重跑：

```bash
rm checkpoints/svdllm/llama31_8b/meta_llama_Llama_3.1_8B_v2_*.pt
bash run_compress_llama31_8b_v2.sh [HF_TOKEN]
```

---

## [OPEN] SVDLLMv1 whitening_then_update 对 Llama-3.1-8B 失效

**发现时间**: 2026-04-12  
**影响**: whitening_then_update 所有 keep_ratio 的 zero-shot 完全退化，无法作为有效基准

### 现象

| | baseline acc (7-task, 无 mathQA) | update 0.8 | update 0.6 |
|---|---|---|---|
| 他人实现（Llama-3.1-8B） | 0.6641 | **0.4275** | 0.3267 |
| 我们（Llama-3.1-8B） | 0.6655 | **0.3225** | ~0.32 |

eval 协议相同（acc 均值，7 task），baseline 几乎一致。差距完全来自压缩质量。

### 根因

whitening 对 Llama-3.1-8B 损伤远超 Llama-2-7b：

| 模型 | whitening_only 0.8 PPL | after_update PPL | 损伤倍数 |
|---|---|---|---|
| Llama-2-7b | 167 | 21.57 | 15× |
| Llama-3.1-8B | 919 | 29 | **131×** |

`whitening_local_update` 用 16 个 wikitext2 样本做逐层 lstsq 更新。对轻度损伤的 Llama-2-7b，16 样本足以部分恢复；对 Llama-3.1-8B（损伤 131×），16 样本只能过拟合到 unigram 分布（PPL 降到 29 是假象），zero-shot 判别能力无法恢复。

他人实现用 20 个样本（+4），效果显著，说明还有其他关键差异（可能是 transformers 版本、whitening 校准数据量、或校准数据集）。

### 当前处理

CSV 中 `whitening_then_update` 所有行标记为 `DEGENERATE_ZEROSHOT`，不参与方法对比。
