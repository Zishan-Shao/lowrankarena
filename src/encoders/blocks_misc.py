"""
Self-contained SVD block implementations using only standard PyTorch ops.
No Triton / CUDA kernel dependencies -- works on CPU and GPU alike.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


class BertLayerShim(nn.Module):
    """Wraps an SVD block so it has the same forward signature as BertLayer."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        raw_mask = attention_mask
        if attention_mask is not None and attention_mask.dim() == 4:
            raw_mask = (attention_mask[:, 0, 0, :] == 0)
        return (self.block(hidden_states, raw_mask),)


class NaiveSVDBlock(nn.Module):
    """Low-rank BERT encoder layer executed with standard PyTorch matmul/einsum.

    Factorises Q, K, V, attention-output projection, and FFN projections using
    caller-supplied factorisation callables (plain SVD, FWSVD, DRONE, ...).

    QKV Mode:
        - "per_head": Per-head factorisation (rank limited to dh=64 for BERT-base)
        - "full": Full-matrix factorisation (paper-style, rank can be 256+)

    Parameter layout matches the existing repo convention so that
    ``FlashSVDBlock`` in flashsvd_backend.py can share the same tensors.
    """

    def __init__(
        self,
        hf_layer,
        rank_attn: int,
        rank_ff: int,
        svd_per_head_fn: Callable,
        svd_low_rank_fn: Callable,
        rank_wo: int = 768,
        qkv_mode: str = "per_head",
        attn_mode: str = "einsum",
    ):
        super().__init__()
        cfg = hf_layer.attention.self
        d_model = cfg.all_head_size
        H = cfg.num_attention_heads
        dh = d_model // H

        # Store for forward
        self.num_heads = H
        self.qkv_mode = qkv_mode
        self.attn_mode = attn_mode  # "einsum" | "sdpa"

        if qkv_mode == "per_head":
            # --- Q / K / V per-head factorisation (original) ---
            WqT = cfg.query.weight.data.t()
            WkT = cfg.key.weight.data.t()
            WvT = cfg.value.weight.data.t()
            bq = cfg.query.bias.data.view(1, H, 1, dh)
            bk = cfg.key.bias.data.view(1, H, 1, dh)
            bv = cfg.value.bias.data.view(1, H, 1, dh)

            Uq, Vq = svd_per_head_fn(WqT, rank_attn)
            Uk, Vk = svd_per_head_fn(WkT, rank_attn)
            Uv, Vv = svd_per_head_fn(WvT, rank_attn)

            # Store as parameters [1, H, dm, R] convention
            self.Pq = nn.Parameter(Uq.unsqueeze(0))
            self.Vq = nn.Parameter(Vq.unsqueeze(0))
            self.bq = nn.Parameter(bq)
            self.Pk = nn.Parameter(Uk.unsqueeze(0))
            self.Vk = nn.Parameter(Vk.unsqueeze(0))
            self.bk = nn.Parameter(bk)
            self.Pv = nn.Parameter(Uv.unsqueeze(0))
            self.Vv = nn.Parameter(Vv.unsqueeze(0))
            self.bv = nn.Parameter(bv)

        elif qkv_mode == "full":
            # --- Q / K / V full-matrix factorisation (paper-style) ---
            Wq = cfg.query.weight.data.t()   # [dm, dm]
            Wk = cfg.key.weight.data.t()
            Wv = cfg.value.weight.data.t()

            bq_full = cfg.query.bias.data    # [dm]
            bk_full = cfg.key.bias.data
            bv_full = cfg.value.bias.data

            Uq, Vq = svd_low_rank_fn(Wq, rank_attn)   # U: [dm, r], V: [r, dm]
            Uk, Vk = svd_low_rank_fn(Wk, rank_attn)
            Uv, Vv = svd_low_rank_fn(Wv, rank_attn)

            self.Uq = nn.Parameter(Uq)
            self.Vq = nn.Parameter(Vq)
            self.bq_full = nn.Parameter(bq_full)

            self.Uk = nn.Parameter(Uk)
            self.Vk = nn.Parameter(Vk)
            self.bk_full = nn.Parameter(bk_full)

            self.Uv = nn.Parameter(Uv)
            self.Vv = nn.Parameter(Vv)
            self.bv_full = nn.Parameter(bv_full)

        else:
            raise ValueError(f"Unknown qkv_mode: {qkv_mode}. Must be 'per_head' or 'full'.")

        # --- FFN factorisation (same for both modes) ---
        Wi = hf_layer.intermediate.dense.weight.data.t()
        bi = hf_layer.intermediate.dense.bias.data
        WoT = hf_layer.output.dense.weight.data.t()
        bo2 = hf_layer.output.dense.bias.data

        U1, V1 = svd_low_rank_fn(Wi, rank_ff)
        U2, V2 = svd_low_rank_fn(WoT, rank_ff)

        # --- Attention output projection (same for both modes) ---
        Wo_full = hf_layer.attention.output.dense.weight.data
        bo_attn = hf_layer.attention.output.dense.bias.data
        Uo, Vo = svd_low_rank_fn(Wo_full.t(), rank_wo)

        self.Uo = nn.Parameter(Uo)
        self.Vo = nn.Parameter(Vo)
        self.bo_attn = nn.Parameter(bo_attn)

        self.U1 = nn.Parameter(U1)
        self.V1 = nn.Parameter(V1)
        self.b1 = nn.Parameter(bi)
        self.U2 = nn.Parameter(U2)
        self.V2 = nn.Parameter(V2)
        self.b2 = nn.Parameter(bo2)

        self.ln1 = hf_layer.attention.output.LayerNorm
        self.ln2 = hf_layer.output.LayerNorm

    # -----------------------------------------------------------------
    def forward(self, x, mask=None):
        B, M, dm = x.shape
        H = self.num_heads
        dh = dm // H
        scale = 1.0 / math.sqrt(dh)

        if self.qkv_mode == "per_head":
            # --- Per-head mode (original) ---
            def project(x, P, V, b):
                tmp = torch.einsum("bmd,hdr->bhmr", x, P)
                return torch.einsum("bhmr,hrd->bhmd", tmp, V) + b

            Q = project(x, self.Pq[0], self.Vq[0], self.bq)
            K = project(x, self.Pk[0], self.Vk[0], self.bk)
            V = project(x, self.Pv[0], self.Vv[0], self.bv)

        elif self.qkv_mode == "full":
            # --- Full-matrix mode (paper-style) ---
            # Q = (x @ Uq) @ Vq + bq_full, then reshape to [B, H, M, dh]
            Q = (x @ self.Uq) @ self.Vq + self.bq_full
            K = (x @ self.Uk) @ self.Vk + self.bk_full
            V = (x @ self.Uv) @ self.Vv + self.bv_full

            # Reshape to multi-head format
            Q = Q.view(B, M, H, dh).transpose(1, 2)  # [B, H, M, dh]
            K = K.view(B, M, H, dh).transpose(1, 2)
            V = V.view(B, M, H, dh).transpose(1, 2)

        # --- Attention (same for both qkv_mode branches) ---
        if self.attn_mode == "sdpa":
            # PyTorch 2.0+ SDPA — does not materialize the full [B,H,M,M] attention matrix.
            # Used for ablation: isolates "the benefit of flash attention itself" from
            # "the extra benefit of FlashSVD Triton kernel fused low-rank projection".
            # Numerically equivalent to the einsum path (absolute logit diff < 1e-5 in fp32).
            if mask is not None:
                sdpa_mask = mask.view(B, 1, 1, M).to(torch.bool)
            else:
                sdpa_mask = None
            attn = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=sdpa_mask, scale=scale, dropout_p=0.0
            )
        else:
            # einsum path — faithfully reproduces the original paper (FWSVD/DRONE/AdaSVD all use this path).
            # Explicitly materializes [B,H,M,M] (O(n²) memory); this is the Naive backend baseline.
            logits = torch.einsum("bhmd,bhnd->bhmn", Q, K) * scale
            if mask is not None:
                m = mask.view(B, 1, 1, M).to(torch.bool)
                logits = logits.masked_fill(~m, torch.finfo(logits.dtype).min)
            A = torch.softmax(logits, dim=-1)
            attn = torch.einsum("bhmn,bhnd->bhmd", A, V)

        # --- Output projection + LN (same for both modes) ---
        attn = attn.transpose(1, 2).reshape(B, M, dm)
        x1 = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn)

        # --- FFN (same for both modes) ---
        mid = x1 @ self.U1
        midV = mid @ self.V1
        midA = F.gelu(midV + self.b1)
        y = (midA @ self.U2) @ self.V2 + self.b2
        return self.ln2(x1 + y)


# ═══════════════════════════════════════════════════════════════════════════
# ModernBERT support (pre-norm, RoPE, GeGLU FFN)
# ═══════════════════════════════════════════════════════════════════════════

def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x, cos, sin):
    return (x * cos) + (_rotate_half(x) * sin)


class ModernBertLayerShim(nn.Module):
    """Wraps an SVD block so it has the same forward signature as a ModernBERT layer."""

    def __init__(self, block: nn.Module, attention_type: str = "global"):
        super().__init__()
        self.block = block
        # transformers >= 4.48 accesses attention_type on each layer during forward
        self.attention_type = getattr(block, "attention_type", attention_type)
        block.attention_type = self.attention_type  # propagate to block for rotary_emb call

    def forward(self, hidden_states, attention_mask=None, sliding_window_mask=None,
                position_ids=None, output_attentions=False, **kwargs):
        return self.block(
            hidden_states,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            **kwargs,
        )


class NaiveModernBertSVDBlock(nn.Module):
    """Low-rank ModernBERT encoder layer with pre-norm, RoPE, and GeGLU FFN.

    Uses standard PyTorch ops: ``F.scaled_dot_product_attention`` for
    attention and explicit matmul chains for the FFN.
    """

    def __init__(
        self,
        hf_layer,
        config,
        rank_attn: int,
        rank_ff: int,
        rank_wo: int,
        svd_per_head_fn: Callable,
        svd_low_rank_fn: Callable,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Preserve model dtype so SVD factors (computed in float32) match activations
        _dt = hf_layer.attn.Wqkv.weight.dtype

        # Pre-norm layers
        self.attn_norm = copy.deepcopy(hf_layer.attn_norm)
        self.mlp_norm = copy.deepcopy(hf_layer.mlp_norm)

        # RoPE (shared reference – not copied)
        # transformers < 4.48: rotary_emb is an nn.Module on attn
        # transformers >= 4.48: rotary_fn is apply_rotary_pos_emb (a plain function, not usable here)
        #   → real ModernBertRotaryEmbedding moved to model.model.rotary_emb; caller must set it after init
        _attn = hf_layer.attn
        _rope = getattr(_attn, "rotary_emb", None)
        self.rotary_emb = _rope if isinstance(_rope, nn.Module) else None

        # --- Split fused Wqkv [3D, D] → Q, K, V ---
        W = hf_layer.attn.Wqkv.weight.data          # [3*D, D]
        b = hf_layer.attn.Wqkv.bias
        Wq, Wk, Wv = torch.chunk(W, 3, dim=0)
        if b is not None:
            bq, bk, bv = torch.chunk(b.data, 3, dim=0)
        else:
            bq = bk = bv = None

        # Per-head SVD: Wq^T [D, D] → Uq [H, D, R], Vq [H, R, dh]
        Uq, Vq = svd_per_head_fn(Wq.t(), rank_attn)
        Uk, Vk = svd_per_head_fn(Wk.t(), rank_attn)
        Uv, Vv = svd_per_head_fn(Wv.t(), rank_attn)

        self.Pq = nn.Parameter(Uq.to(_dt))   # [H, D, R]
        self.Vq = nn.Parameter(Vq.to(_dt))   # [H, R, dh]
        self.Pk = nn.Parameter(Uk.to(_dt))
        self.Vk = nn.Parameter(Vk.to(_dt))
        self.Pv = nn.Parameter(Uv.to(_dt))
        self.Vv = nn.Parameter(Vv.to(_dt))

        self.bq = nn.Parameter(bq.clone().to(_dt)) if bq is not None else None
        self.bk = nn.Parameter(bk.clone().to(_dt)) if bk is not None else None
        self.bv = nn.Parameter(bv.clone().to(_dt)) if bv is not None else None

        # Attention output projection (SVD factorized, same as BERT Wo)
        Wo_a = hf_layer.attn.Wo
        U_wo, V_wo = svd_low_rank_fn(Wo_a.weight.data.t(), rank_wo)
        self.Uo = nn.Parameter(U_wo.to(_dt))   # [D, rank_wo]
        self.Vo = nn.Parameter(V_wo.to(_dt))   # [rank_wo, D]
        self.bo_attn = nn.Parameter(Wo_a.bias.data.clone().to(_dt)) if Wo_a.bias is not None else None

        # --- FFN: Wi [2*D_ffn, D] (GeGLU), Wo [D, D_ffn] ---
        Wi = hf_layer.mlp.Wi
        Wo = hf_layer.mlp.Wo
        U1, V1 = svd_low_rank_fn(Wi.weight.data.t(), rank_ff)
        U2, V2 = svd_low_rank_fn(Wo.weight.data.t(), rank_ff)

        self.U1 = nn.Parameter(U1.to(_dt))
        self.V1 = nn.Parameter(V1.to(_dt))
        self.b1 = nn.Parameter(Wi.bias.data.clone().to(_dt)) if Wi.bias is not None else None
        self.U2 = nn.Parameter(U2.to(_dt))
        self.V2 = nn.Parameter(V2.to(_dt))
        self.b2 = nn.Parameter(Wo.bias.data.clone().to(_dt)) if Wo.bias is not None else None

        # GeGLU detection
        self.ffn_D = Wo.in_features
        self.ffn_is_geglu = (Wi.out_features == 2 * self.ffn_D)
        gelu_approx = getattr(
            getattr(hf_layer.mlp, "act", nn.GELU()), "approximate", "tanh"
        )
        self.gelu_approximate = gelu_approx

    # -----------------------------------------------------------------
    def forward(self, hidden_states, attention_mask=None,
                sliding_window_mask=None, position_ids=None,
                output_attentions=False, **kwargs):
        B, M, D = hidden_states.shape
        H, dh = self.num_heads, self.head_dim

        # === Attention (pre-norm) ===
        x = hidden_states
        xn = self.attn_norm(x)

        def project(xn, P, V, b):
            tmp = torch.einsum("bmd,hdr->bhmr", xn, P.to(xn.dtype))
            out = torch.einsum("bhmr,hrd->bhmd", tmp, V.to(xn.dtype))
            if b is not None:
                out = out + b.to(xn.dtype).view(1, H, 1, dh)
            return out

        Q = project(xn, self.Pq, self.Vq, self.bq)
        K = project(xn, self.Pk, self.Vk, self.bk)
        V = project(xn, self.Pv, self.Vv, self.bv)

        # RoPE
        if position_ids is None:
            position_ids = torch.arange(M, device=x.device).unsqueeze(0).expand(B, M)
        qf = Q.reshape(B * H, M, dh)
        kf = K.reshape(B * H, M, dh)
        posf = position_ids.unsqueeze(1).expand(B, H, M).reshape(B * H, M)
        try:
            cos, sin = self.rotary_emb(qf, position_ids=posf, layer_type=getattr(self, 'attention_type', 'global'))
        except TypeError:
            cos, sin = self.rotary_emb(qf, position_ids=posf)
        Q = _apply_rotary(qf, cos, sin).view(B, H, M, dh)
        K = _apply_rotary(kf, cos, sin).view(B, H, M, dh)

        # SDPA mask
        sdpa_mask = None
        if sliding_window_mask is not None:
            sm = sliding_window_mask
            if sm.dtype.is_floating_point and sm.dtype != Q.dtype:
                sm = sm.to(Q.dtype)
            sdpa_mask = sm
        elif attention_mask is not None:
            if attention_mask.dim() == 2:
                sdpa_mask = ~(attention_mask.to(torch.bool))[:, None, None, :]
            elif attention_mask.dim() == 4:
                sm = attention_mask
                if sm.dtype.is_floating_point and sm.dtype != Q.dtype:
                    sm = sm.to(Q.dtype)
                sdpa_mask = sm

        attn = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=sdpa_mask, dropout_p=0.0,
        )
        attn = attn.transpose(1, 2).reshape(B, M, D)
        attn_out = (attn @ self.Uo) @ self.Vo
        if self.bo_attn is not None:
            attn_out = attn_out + self.bo_attn
        x = x + attn_out

        # === FFN (pre-norm, GeGLU) ===
        xn2 = self.mlp_norm(x)
        z = (xn2 @ self.U1) @ self.V1
        if self.b1 is not None:
            z = z + self.b1

        if self.ffn_is_geglu:
            z1, z2 = z.chunk(2, dim=-1)
            h = F.gelu(z1, approximate=self.gelu_approximate) * z2
        else:
            h = F.gelu(z, approximate=self.gelu_approximate)

        y = (h @ self.U2) @ self.V2
        if self.b2 is not None:
            y = y + self.b2
        x = x + y

        if output_attentions:
            return (x, None)
        return x
