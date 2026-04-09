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

## [OPEN] SVDLLMv2 zero-shot 大面积退化

**发现时间**: 2026-04-07  
**关联**: 同上

SVDLLMv2（hf_v2_*）所有 keep_ratio 的 boolq=0.378287、mathqa=0.205695，但 wiki2 PPL 从 1180（0.4）到 28100（0.8）不等。PPL 偏高且 zero-shot 退化，说明 v2 在当前压缩参数下质量较差，或与 whitening_then_update 同一根因。待服务器验证。
