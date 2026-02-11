# SVD-LLM v1/v2 应用到 BERT Family 的设计方案

## 1. SVD-LLM 方法概述

### 1.1 SVD-LLM v1 (Original)
**来源**: `baselines/SVD-LLM/flashsvd_component/svd_llama.py`

**Rank 计算公式**:
- **MLP层 (FFN)**: `R = int(m × n × ratio / (m + n))`
  - m = hidden_size (输入维度)
  - n = intermediate_size (中间层维度)
  - 这是**调和平均**的变体，偏向于较小维度

- **Attention层**: `R = int(hidden_size × ratio / 2)`
  - 注意这里除以2，是因为QKV共享相同的低秩
  - 对于output projection也使用相同的rank

**特点**:
- 非对称设计，考虑了输入输出维度的不对称性
- Attention使用固定的/2因子，较为保守
- 适合decoder架构（MLP通常m << n，如4096 vs 11008）

### 1.2 SVD-LLM v2 (Unified)
**来源**: `baselines/SVD-LLM/component/svd_llama.py` (compat_ranks=False)

**Rank 计算公式**:
- **统一公式**: `R = int(min(m, n) × ratio)`
  - 对MLP和Attention都使用相同的公式
  - 直接按最小维度的比例分配rank

**特点**:
- 对称设计，简单统一
- Attention层相比v1提高了2倍rank (去掉了/2)
- 更激进的压缩策略（相同ratio下rank更大）

---

## 2. BERT Family 架构分析

### 2.1 BERT & RoBERTa
**模型路径**: `model.bert.encoder.layer[i]` (BERT) / `model.roberta.encoder.layer[i]` (RoBERTa)

**架构特点**:
- **Attention**:
  - Post-norm (LayerNorm在残差之后)
  - Absolute positional embeddings
  - Fused QKV projection: `nn.Linear(hidden, 3*hidden, bias=True)`
  - Separate Q/K/V: `self.query`, `self.key`, `self.value`
  - Output: `self.output.dense` (hidden → hidden)

- **MLP (FFN)**:
  - Up projection: `self.intermediate.dense` (hidden → intermediate)
  - Activation: GELU
  - Down projection: `self.output.dense` (intermediate → hidden)

- **典型尺寸** (bert-base):
  - hidden_size = 768
  - intermediate_size = 3072
  - num_attention_heads = 12
  - head_dim = 64

### 2.2 ModernBERT
**模型路径**: `model.model.layers[i]`

**架构特点**:
- **Attention**:
  - Pre-norm (LayerNorm在残差之前)
  - RoPE (Rotary Position Embeddings)
  - Fused Wqkv: `self.attn.Wqkv` (hidden → 3*hidden)
  - Output: `self.attn.Wo` (hidden → hidden)

- **MLP (FFN)**:
  - **GeGLU** (Gated Linear Unit with GeLU)
  - Wi (gate & up): `self.mlp.Wi` (hidden → 2*intermediate)
  - Wo (down): `self.mlp.Wo` (intermediate → hidden)
  - `output = Wo(GeLU(gate) * up)`

- **典型尺寸** (modernbert-base):
  - hidden_size = 768
  - intermediate_size = 1152
  - num_attention_heads = 12
  - head_dim = 64

---

## 3. Rank 计算对比分析

### 3.1 BERT/RoBERTa (bert-base: H=768, I=3072)

#### MLP层
| Ratio | v1 Formula | v1 Rank | v2 Formula | v2 Rank | 参数量对比 |
|-------|------------|---------|------------|---------|-----------|
| 0.3 | 768×3072×0.3/(768+3072) | **184** | min(768,3072)×0.3 | **230** | v2多25% |
| 0.5 | 768×3072×0.5/(768+3072) | **307** | min(768,3072)×0.5 | **384** | v2多25% |
| 0.7 | 768×3072×0.7/(768+3072) | **430** | min(768,3072)×0.7 | **537** | v2多25% |

**原始参数量**: 768×3072 = 2,359,296
**SVD参数量**: (768+3072)×R = 3840×R
- v1 @ ratio=0.3: 3840×184 = 706,560 (29.9%)
- v2 @ ratio=0.3: 3840×230 = 883,200 (37.4%)

#### Attention层 (QKV + O)
| Ratio | v1 Formula (Q/K/V) | v1 Rank | v2 Formula | v2 Rank | 说明 |
|-------|-------------------|---------|------------|---------|------|
| 0.3 | 768×0.3/2 | **115** | 768×0.3 | **230** | v2是v1的2倍 |
| 0.5 | 768×0.5/2 | **192** | 768×0.5 | **384** | v2是v1的2倍 |
| 0.7 | 768×0.7/2 | **268** | 768×0.7 | **537** | v2是v1的2倍 |

**原始参数量**: 4×768×768 = 2,359,296 (QKV + O)
**SVD参数量**: 4×(768+768)×R = 6144×R
- v1 @ ratio=0.3: 6144×115 = 706,560 (29.9%)
- v2 @ ratio=0.3: 6144×230 = 1,413,120 (59.9%)

**关键差异**: v2的Attention层压缩比实际上比v1宽松得多！

### 3.2 ModernBERT (modernbert-base: H=768, I=1152)

#### MLP层 (GeGLU)
| Ratio | v1 Formula | v1 Rank | v2 Formula | v2 Rank | 参数量对比 |
|-------|------------|---------|------------|---------|-----------|
| 0.3 | 768×1152×0.3/(768+1152) | **138** | min(768,1152)×0.3 | **230** | v2多67% |
| 0.5 | 768×1152×0.5/(768+1152) | **230** | min(768,1152)×0.5 | **384** | v2多67% |
| 0.7 | 768×1152×0.7/(768+1152) | **322** | min(768,1152)×0.7 | **537** | v2多67% |

**特殊性**: ModernBERT的intermediate_size更小(1152 vs 3072)，v1和v2的差异更大

#### Attention层 (同BERT)
与BERT/RoBERTa相同，v2 rank是v1的2倍

---

## 4. 应用到 BERT Family 的设计方案

### 4.1 整体架构设计

```
BERTSVDLLMv1/v2
├── SVDAttentionBlock
│   ├── Q, K, V 低秩分解: W = U @ V (R由v1/v2公式决定)
│   ├── O 低秩分解
│   └── 保留原始架构特性 (post-norm/pre-norm, RoPE/abs-pos)
└── SVDMLPBlock
    ├── Up/Gate 低秩分解
    ├── Down 低秩分解
    └── 保留原始激活函数 (GELU/GeGLU)
```

### 4.2 BERT/RoBERTa 适配方案

#### 方案A: 独立 Q/K/V (与BERTWhiten一致)
**优点**:
- 每个projection独立SVD，灵活性高
- 可以对Q/K/V使用不同的rank（如AdaSVD）
- 与现有BERTWhiten代码结构一致

**缺点**:
- 需要修改原始模型结构（拆分Wqkv）
- 无法直接使用FlashSVD的fused kernel

**实现**:
```
# 伪代码结构
class SVDLLMBertAttention:
    q_u, q_v = low_rank_decompose(query.weight, rank_q)
    k_u, k_v = low_rank_decompose(key.weight, rank_k)
    v_u, v_v = low_rank_decompose(value.weight, rank_v)
    o_u, o_v = low_rank_decompose(output.dense.weight, rank_o)

    forward:
        Q = q_u @ q_v @ x
        K = k_u @ k_v @ x
        V = v_u @ v_v @ x
        attn_out = attention(Q, K, V)
        return o_u @ o_v @ attn_out
```

#### 方案B: Fused QKV (适用FlashSVD)
**优点**:
- 可以使用FlashSVD加速（if adapted to BERT）
- 保持原始fused结构

**缺点**:
- Q/K/V必须使用相同rank
- 需要特殊的tensor layout处理

### 4.3 ModernBERT 适配方案

**关键挑战**:
1. **RoPE**: 需要在SVD之后应用rotary embeddings
2. **GeGLU**: gate和up是fused的 (Wi: hidden → 2*intermediate)
3. **Pre-norm**: 归一化在projection之前

**实现策略**:
```
# 伪代码
class SVDLLMModernBertAttention:
    # Wqkv: [hidden, 3*hidden] -> 分解为3个独立部分
    qkv_u, qkv_v = low_rank_decompose_fused(Wqkv, rank)
    wo_u, wo_v = low_rank_decompose(Wo, rank)

    forward:
        P = qkv_v @ x  # [B, L, R]
        QKV = qkv_u @ P  # [B, L, 3*hidden]
        Q, K, V = split(QKV, 3)
        Q, K = apply_rope(Q, K)  # RoPE AFTER low-rank
        attn_out = attention(Q, K, V)
        return wo_u @ wo_v @ attn_out

class SVDLLMModernBertMLP:
    # Wi: [hidden, 2*intermediate] -> gate + up
    wi_u, wi_v = low_rank_decompose(Wi, rank)
    wo_u, wo_v = low_rank_decompose(Wo, rank)

    forward:
        P = wi_v @ x  # [B, L, R]
        GU = wi_u @ P  # [B, L, 2*intermediate]
        gate, up = split(GU, 2)
        return wo_u @ wo_v @ (gelu(gate) * up)
```

---

## 5. Rank Selection Strategy 设计

### 5.1 统一 Rank (推荐用于FlashSVD)
**适用场景**: 需要使用FlashSVD kernel加速

**策略**:
- 所有层使用相同的ratio
- Attention层: Q/K/V/O使用相同rank
- MLP层: up/gate/down使用相同rank (或MLP与Attn不同)

**示例配置**:
```python
config = {
    "version": "v1",  # or "v2"
    "ratio": 0.5,
    "unified_rank": True,
    "attn_rank": None,  # auto-computed from ratio
    "mlp_rank": None,   # auto-computed from ratio
}
```

### 5.2 Per-Op Rank (推荐用于Naive backend)
**适用场景**: 追求最优压缩率，可以接受慢速

**策略**:
- 每个linear projection独立计算rank
- 可以结合AdaSVD进行自适应rank选择
- Q/K/V可以有不同rank

**示例配置**:
```python
config = {
    "version": "v2",
    "ratio": 0.5,
    "unified_rank": False,
    "per_layer_ratio": {
        0: 0.4,  # layer 0 more aggressive
        1: 0.5,
        # ...
    }
}
```

### 5.3 Hybrid Strategy
**策略**:
- Attention层使用v1公式（保守，保护关键信息）
- MLP层使用v2公式（激进，MLP冗余度高）

---

## 6. 实现优先级建议

### Phase 1: 基础实现 (Naive Backend)
**目标**: 验证v1/v2方法的有效性

**实现**:
1. **BertSVDLLMv1/v2**: 基于BERT的独立Q/K/V实现
   - 复用BERTWhiten的block结构
   - 替换rank计算逻辑为v1/v2公式
   - 支持post-norm + GELU

2. **RoBertaSVDLLMv1/v2**: 直接复用BertSVDLLM
   - RoBERTa与BERT结构完全相同
   - 唯一区别是模型路径前缀 (roberta vs bert)

3. **测试**: 在SST-2上验证v1 vs v2的性能差异

### Phase 2: ModernBERT支持
**目标**: 扩展到pre-norm + RoPE + GeGLU架构

**实现**:
1. **ModernBertSVDLLMv1/v2**:
   - 处理RoPE的低秩适配
   - 处理GeGLU的fused gate/up
   - 测试pre-norm的影响

2. **测试**: 对比BERT vs ModernBERT的压缩效率差异

### Phase 3: FlashSVD加速 (Optional)
**目标**: 集成FlashSVD kernel加速

**前提条件**:
- Phase 1/2验证v1/v2方法有效
- 决定使用unified rank还是per-op rank

**实现**:
1. 适配BERT架构到FlashSVD kernel
2. 处理post-norm vs pre-norm的差异
3. 集成到eval_encoder benchmark

---

## 7. 关键设计决策

### 7.1 是否添加Whiten/DRONE方法？

**分析**:
- **SVD-LLM原始实现**: 不包含Whiten/DRONE，只使用标准SVD
- **BERTWhiten实现**: 使用DRONE (data-aware covariance calibration)

**建议**:
- **第一阶段**: 不添加Whiten，与SVD-LLM baseline保持一致
- **第二阶段**: 可以作为optional feature添加（参考BERTWhiten）
- **命名**:
  - `BertSVDLLMv1/v2`: 标准SVD
  - `BertSVDLLMv1/v2_Whiten`: 添加DRONE calibration

### 7.2 v1 vs v2 哪个更好？

**预测**:
- **BERT/RoBERTa**:
  - v1可能更适合（MLP维度差异大，3072 vs 768）
  - v2的Attention层rank过大，可能浪费参数

- **ModernBERT**:
  - v2可能更适合（intermediate_size较小，1152 vs 768）
  - v1的MLP rank可能过小

**实验验证**: 需要在多个ratio (0.3, 0.5, 0.7)下对比测试

### 7.3 是否支持GQA (Grouped Query Attention)？

**当前状态**:
- SVD-LLM v2支持GQA (num_key_value_heads < num_heads)
- BERT family不使用GQA

**建议**: 第一阶段不实现，因为BERT不需要

---

## 8. 预期的参数压缩率对比

### BERT-base (H=768, 12层)
**原始参数量**: ~110M

| Method | Ratio | 每层MLP | 每层Attn | 总压缩 | 预估大小 |
|--------|-------|---------|----------|--------|----------|
| v1 | 0.3 | 29.9% | 29.9% | ~30% | ~33M |
| v2 | 0.3 | 37.4% | 59.9% | ~49% | ~54M |
| v1 | 0.5 | 49.8% | 49.8% | ~50% | ~55M |
| v2 | 0.5 | 62.3% | 99.8% | ~81% | ~89M |

**结论**: v2在相同ratio下保留更多参数，尤其是Attention层

---

## 9. 下一步行动

1. **Phase 1实现**: 先实现BertSVDLLMv1作为baseline
2. **对比测试**: v1 vs v2 在BERT-base SST-2上的性能
3. **Rank分析**: 可视化不同ratio下的rank分布和性能曲线
4. **扩展**: 逐步添加RoBERTa和ModernBERT支持

---

## 10. 参考实现路径

- **Baseline v1**: `baselines/SVD-LLM/flashsvd_component/svd_llama.py`
- **Baseline v2**: `baselines/SVD-LLM/component/svd_llama.py`
- **BERT Whiten**: `src/encoders/BERTWhiten/profile_svd.py`
- **Decoder实现**: `src/decoders/SVDLLM/svd_llama.py`
