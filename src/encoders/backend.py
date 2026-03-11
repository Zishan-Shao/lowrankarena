"""
FlashSVD backend: replaces NaiveSVDBlock forward passes with Triton-accelerated
flash_svd_attention + flashsvd_ffn kernels.

Usage:
    from src.encoders.backend import enable_flashsvd
    enable_flashsvd(model)          # in-place, raises if kernels unavailable
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.blocks_misc import NaiveSVDBlock, BertLayerShim, NaiveModernBertSVDBlock

# Import MinimalSVDBlock / MinimalModernBertSVDBlock if available (for checkpoint loading)
try:
    from src.encoders.io import MinimalSVDBlock
except ImportError:
    MinimalSVDBlock = None

try:
    from src.encoders.io import MinimalModernBertSVDBlock
except ImportError:
    MinimalModernBertSVDBlock = None

# ---------------------------------------------------------------------------
# Lazy kernel imports – fail fast with a human-readable message
# ---------------------------------------------------------------------------
_flash_svd_attention = None
_flashsvd_ffn_v1 = None
_flash_svd_attention_v15 = None
_flashsvd_ffn_v15_fn = None
_flashsvd_ffn_geglu_v15_fn = None


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n (required by tl.arange in Triton kernels)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _resolve_kernel_dir():
    """Return the encoder-kernel directory path."""
    local_kernel_dir = os.path.join(os.path.dirname(__file__), "kernels")
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    repo_kernel_dir = os.path.join(_REPO_ROOT, "src", "kernels", "encoder")
    if os.path.isdir(local_kernel_dir):
        return local_kernel_dir
    elif os.path.isdir(repo_kernel_dir):
        return repo_kernel_dir
    else:
        raise RuntimeError(
            f"FlashSVD kernel directory not found.\n"
            f"Tried: {local_kernel_dir} and {repo_kernel_dir}\n"
            "Make sure kernels are available."
        )


def _import_kernels():
    global _flash_svd_attention, _flashsvd_ffn_v1
    if _flash_svd_attention is not None:
        return  # already imported

    kernel_dir = _resolve_kernel_dir()
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)

    try:
        from flashsvdattn import flash_svd_attention
        from flashsvdffnv1 import flashsvd_ffn_v1
    except ImportError as e:
        raise RuntimeError(
            "Cannot import FlashSVD Triton kernels. "
            "Ensure Triton is installed (`pip install triton`) and a CUDA GPU "
            f"is available.\nOriginal error: {e}"
        ) from e

    _flash_svd_attention = flash_svd_attention
    _flashsvd_ffn_v1 = flashsvd_ffn_v1


def _import_kernels_v15():
    global _flash_svd_attention_v15, _flashsvd_ffn_v15_fn
    if _flash_svd_attention_v15 is not None:
        return  # already imported

    kernel_dir = _resolve_kernel_dir()
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)

    try:
        from flashsvdattn15 import flash_svd_attention_v15
        from flashsvdffnv15 import flashsvd_ffn_v15
    except ImportError as e:
        raise RuntimeError(
            "Cannot import FlashSVD v1.5 Triton kernels. "
            "Ensure Triton is installed (`pip install triton`) and a CUDA GPU "
            f"is available.\nOriginal error: {e}"
        ) from e

    _flash_svd_attention_v15 = flash_svd_attention_v15
    _flashsvd_ffn_v15_fn = flashsvd_ffn_v15


def _import_kernels_modernbert():
    """Import FlashSVD GeGLU v1.5 kernel for ModernBERT FFN."""
    global _flashsvd_ffn_geglu_v15_fn
    if _flashsvd_ffn_geglu_v15_fn is not None:
        return

    kernel_dir = _resolve_kernel_dir()
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)

    try:
        from flashsvdgeglu_v15 import flashsvd_ffn_geglu_autotuned
    except ImportError as e:
        raise RuntimeError(
            "Cannot import FlashSVD GeGLU v1.5 kernel for ModernBERT. "
            "Ensure Triton is installed and a CUDA GPU is available.\n"
            f"Original error: {e}"
        ) from e

    _flashsvd_ffn_geglu_v15_fn = flashsvd_ffn_geglu_autotuned


# ---------------------------------------------------------------------------
# FlashSVDBlock – same parameters as NaiveSVDBlock, Triton-accelerated forward
# ---------------------------------------------------------------------------
class FlashSVDBlock(nn.Module):
    """Drop-in replacement for NaiveSVDBlock that uses FlashSVD Triton kernels.

    Parameters are *shared* (not copied) from the source NaiveSVDBlock so that
    weight identity is guaranteed between naive and flash runs.
    """

    def __init__(self, naive_block: NaiveSVDBlock):
        super().__init__()
        _import_kernels()

        # Share every parameter tensor (no copy)
        self.Pq = naive_block.Pq
        self.Vq = naive_block.Vq
        self.bq = naive_block.bq
        self.Pk = naive_block.Pk
        self.Vk = naive_block.Vk
        self.bk = naive_block.bk
        self.Pv = naive_block.Pv
        self.Vv = naive_block.Vv
        self.bv = naive_block.bv

        self.Uo = naive_block.Uo
        self.Vo = naive_block.Vo
        self.bo_attn = naive_block.bo_attn

        self.U1 = naive_block.U1
        self.V1 = naive_block.V1
        self.b1 = naive_block.b1
        self.U2 = naive_block.U2
        self.V2 = naive_block.V2
        self.b2 = naive_block.b2

        self.ln1 = naive_block.ln1
        self.ln2 = naive_block.ln2

        # Pre-squeeze bias from [1, H, 1, dh] → [1, H, dh] for kernel compat
        self._bq_sq = nn.Parameter(naive_block.bq.data.squeeze(2))
        self._bk_sq = nn.Parameter(naive_block.bk.data.squeeze(2))
        self._bv_sq = nn.Parameter(naive_block.bv.data.squeeze(2))

    # -----------------------------------------------------------------
    def forward(self, x, mask=None):
        B, M, dm = x.shape
        # Pq stored as [1, H, dm, R] (NaiveSVDBlock) or [H, dm, R] (profile_flashsvd checkpoint)
        Pq = self.Pq[0] if self.Pq.ndim == 4 else self.Pq  # [H, dm, R]
        Pk = self.Pk[0] if self.Pk.ndim == 4 else self.Pk
        Pv = self.Pv[0] if self.Pv.ndim == 4 else self.Pv
        H = Pq.shape[0]
        R = Pq.shape[2]
        dh = dm // H

        # --- project x into low-rank space per head ---
        tmp_q = torch.einsum("bmd,hdr->bhmr", x, Pq).contiguous()
        tmp_k = torch.einsum("bmd,hdr->bhmr", x, Pk).contiguous()
        tmp_v = torch.einsum("bmd,hdr->bhmr", x, Pv).contiguous()

        # Vq stored as [1, H, R, dh] (NaiveSVDBlock) or [H, R, dh] (profile_flashsvd checkpoint)
        Vq_b = self.Vq[0] if self.Vq.ndim == 4 else self.Vq  # [H, R, dh]
        Vk_b = self.Vk[0] if self.Vk.ndim == 4 else self.Vk
        Vv_b = self.Vv[0] if self.Vv.ndim == 4 else self.Vv

        # expand V / bias to [B, H, ...] for the Triton kernel
        Vq_f = Vq_b.expand(B, H, R, dh)
        Vk_f = Vk_b.expand(B, H, R, dh)
        Vv_f = Vv_b.expand(B, H, R, dh)
        bq_f = self._bq_sq.expand(B, H, dh)
        bk_f = self._bk_sq.expand(B, H, dh)
        bv_f = self._bv_sq.expand(B, H, dh)

        # --- flash SVD attention ---
        # block_m must match kernels/encoder_kernels/flashsvdattn.py BLOCK_M=64.
        # Passing block_m=32 would compute grid with 32-row tiles but the Triton
        # kernel processes 64-row tiles → out-of-bounds reads → CUDA illegal access.
        if mask is not None:
            mask4 = mask.view(B, 1, 1, M)
        else:
            # No padding mask → all tokens valid (full attention)
            mask4 = torch.ones(B, 1, 1, M, dtype=torch.bool, device=x.device)
        attn_out = _flash_svd_attention(
            tmp_q, Vq_f, bq_f,
            tmp_k, Vk_f, bk_f,
            tmp_v, Vv_f, bv_f,
            mask=mask4,
            block_r=R,
        )
        del tmp_q, tmp_k, tmp_v, Vq_f, Vk_f, Vv_f, bq_f, bk_f, bv_f

        attn = attn_out.view(B, H, M, dh).transpose(1, 2).reshape(B, M, dm)
        x1 = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn)

        # --- flash SVD FFN ---
        mid = x1 @ self.U1
        y = _flashsvd_ffn_v1(mid, self.V1, self.U2, self.V2, self.b1, self.b2)
        return self.ln2(x1 + y)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def enable_flashsvd(model: nn.Module) -> nn.Module:
    """Patch a model in-place: swap every NaiveSVDBlock for a FlashSVDBlock.

    Parameters
    ----------
    model : nn.Module
        A ``BertForSequenceClassification`` (or similar) whose encoder layers
        have been wrapped with ``BertLayerShim(NaiveSVDBlock(...))``.

    Returns
    -------
    model : nn.Module   (same object, mutated in-place)

    Raises
    ------
    RuntimeError
        If Triton / FlashSVD kernels cannot be imported, or if no SVD blocks
        are found in the model.
    """
    _import_kernels()  # fail-fast if kernels are missing

    # Locate encoder layers
    model_type = getattr(model.config, "model_type", "").lower()
    encoder_layers = None
    if model_type == "modernbert":
        raise RuntimeError(
            "enable_flashsvd: ModernBERT is not supported by FlashSVD v1. "
            "Use --backend flashsvd15 (fused GeGLU kernel) or --backend naive."
        )
    elif hasattr(model, "bert"):
        encoder_layers = model.bert.encoder.layer
    elif hasattr(model, "roberta"):
        encoder_layers = model.roberta.encoder.layer
    else:
        raise RuntimeError(
            "enable_flashsvd: cannot find .bert.encoder.layer or "
            ".roberta.encoder.layer on the supplied model."
        )

    patched = 0
    already_flash = 0
    for i, layer in enumerate(encoder_layers):
        block = getattr(layer, "block", None)
        _cls_name = type(block).__name__ if block is not None else ""

        # Idempotency: skip layers that are already FlashSVDBlock
        if _cls_name == "FlashSVDBlock" or isinstance(block, FlashSVDBlock):
            already_flash += 1
            continue

        # Support both NaiveSVDBlock (from compression) and MinimalSVDBlock (from checkpoint loading).
        # isinstance() may fail if adasvd_wrapper imported blocks via a different sys.path entry
        # (module aliasing: "blocks.NaiveSVDBlock" vs "eval_encoder.blocks.NaiveSVDBlock").
        # Duck-typing fallback ensures correctness regardless of import path.
        is_svd_block = (
            isinstance(block, NaiveSVDBlock)
            or (MinimalSVDBlock and isinstance(block, MinimalSVDBlock))
            or _cls_name in ("NaiveSVDBlock", "MinimalSVDBlock")
        )
        if is_svd_block:
            flash_block = FlashSVDBlock(block)
            layer.block = flash_block
            patched += 1

    if patched == 0 and already_flash == 0:
        raise RuntimeError(
            "enable_flashsvd: no NaiveSVDBlock or MinimalSVDBlock instances found. "
            "Did you run compression (--method != dense) before calling "
            "enable_flashsvd?"
        )

    if already_flash > 0 and patched == 0:
        print(f"[flashsvd] Already enabled ({already_flash} layers) — no-op.")
    else:
        print(f"[flashsvd] Patched {patched} encoder layers with FlashSVD kernels.")
    return model


# ---------------------------------------------------------------------------
# FlashSVD15Block – v1.5 kernels (rank-space kernel, native fp16/bf16, no expand)
# ---------------------------------------------------------------------------
class FlashSVD15Block(nn.Module):
    """Drop-in replacement for NaiveSVDBlock using FlashSVD v1.5 Triton kernels.

    Key differences from FlashSVDBlock (v1):
    - Attention: passes V/bias as [H,R,D]/[H,D] directly (no batch-dim expand)
    - FFN: uses flashsvd_ffn_v15 with native fp16/bf16 support and fp32 auto-cast

    Parameters are *shared* (not copied) from the source NaiveSVDBlock.
    """

    def __init__(self, naive_block: NaiveSVDBlock):
        super().__init__()
        _import_kernels_v15()

        # Share every parameter tensor (no copy)
        self.Pq = naive_block.Pq
        self.Vq = naive_block.Vq
        self.bq = naive_block.bq
        self.Pk = naive_block.Pk
        self.Vk = naive_block.Vk
        self.bk = naive_block.bk
        self.Pv = naive_block.Pv
        self.Vv = naive_block.Vv
        self.bv = naive_block.bv

        self.Uo = naive_block.Uo
        self.Vo = naive_block.Vo
        self.bo_attn = naive_block.bo_attn

        self.U1 = naive_block.U1
        self.V1 = naive_block.V1
        self.b1 = naive_block.b1
        self.U2 = naive_block.U2
        self.V2 = naive_block.V2
        self.b2 = naive_block.b2

        self.ln1 = naive_block.ln1
        self.ln2 = naive_block.ln2

        # Pre-squeeze bias from [1, H, 1, dh] → [1, H, dh] for kernel compat
        # v1.5 kernel auto-strips the leading 1 for bq/bv; bk is softmax-invariant.
        self._bq_sq = nn.Parameter(naive_block.bq.data.squeeze(2))
        self._bk_sq = nn.Parameter(naive_block.bk.data.squeeze(2))
        self._bv_sq = nn.Parameter(naive_block.bv.data.squeeze(2))

    # -----------------------------------------------------------------
    def forward(self, x, mask=None):
        B, M, dm = x.shape
        # Pq stored as [1, H, dm, R] (NaiveSVDBlock) or [H, dm, R] (checkpoint)
        Pq = self.Pq[0] if self.Pq.ndim == 4 else self.Pq  # [H, dm, R]
        Pk = self.Pk[0] if self.Pk.ndim == 4 else self.Pk
        Pv = self.Pv[0] if self.Pv.ndim == 4 else self.Pv
        H = Pq.shape[0]
        R = Pq.shape[2]

        # --- project x into low-rank space per head ---
        tmp_q = torch.einsum("bmd,hdr->bhmr", x, Pq).contiguous()
        tmp_k = torch.einsum("bmd,hdr->bhmr", x, Pk).contiguous()
        tmp_v = torch.einsum("bmd,hdr->bhmr", x, Pv).contiguous()

        # Vq stored as [1, H, R, dh] or [H, R, dh] — v1.5 handles both
        Vq_b = self.Vq[0] if self.Vq.ndim == 4 else self.Vq  # [H, R, dh]
        Vk_b = self.Vk[0] if self.Vk.ndim == 4 else self.Vk
        Vv_b = self.Vv[0] if self.Vv.ndim == 4 else self.Vv

        # v1.5: pass V/bias without batch expansion; kernel accepts [H,R,D] and [1,H,D]
        if mask is not None:
            mask4 = mask.view(B, 1, 1, M)
        else:
            mask4 = torch.ones(B, 1, 1, M, dtype=torch.bool, device=x.device)

        # Triton tl.arange(0, BLOCK_R) requires BLOCK_R = R to be a power of 2.
        # Pad the rank dimension to next_pow2(R) with zeros (no-op when R is already pow2).
        R_pad = _next_pow2(R)
        if R_pad != R:
            pad = R_pad - R
            tmp_q = F.pad(tmp_q, (0, pad))   # [B,H,M,R] → [B,H,M,R_pad]
            tmp_k = F.pad(tmp_k, (0, pad))
            tmp_v = F.pad(tmp_v, (0, pad))
            Vq_b = F.pad(Vq_b, (0, 0, 0, pad))  # [H,R,dh] → [H,R_pad,dh]
            Vk_b = F.pad(Vk_b, (0, 0, 0, pad))
            Vv_b = F.pad(Vv_b, (0, 0, 0, pad))

        attn_out = _flash_svd_attention_v15(
            tmp_q, Vq_b, self._bq_sq,
            tmp_k, Vk_b, self._bk_sq,
            tmp_v, Vv_b, self._bv_sq,
            mask=mask4,
            block_r=R_pad,
        )
        del tmp_q, tmp_k, tmp_v

        dh = dm // H
        attn = attn_out.view(B, H, M, dh).transpose(1, 2).reshape(B, M, dm)
        x1 = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn)

        # --- FlashSVD v1.5 FFN (native fp16/bf16; fp32 auto-cast fallback) ---
        mid = x1 @ self.U1
        y = _flashsvd_ffn_v15_fn(mid, self.V1, self.U2, self.V2, self.b1, self.b2)
        return self.ln2(x1 + y)


# ---------------------------------------------------------------------------
# FlashModernBertSVDBlock – ModernBERT: fused GeGLU FFN kernel, SDPA attention
# ---------------------------------------------------------------------------
class FlashModernBertSVDBlock(nn.Module):
    """Drop-in replacement for NaiveModernBertSVDBlock.

    Attention: unchanged (per-head einsum projection + PyTorch SDPA + RoPE).
    FFN: replaced with fused flashsvd_ffn_geglu_autotuned Triton kernel,
         which avoids materialising the intermediate S buffer.

    Parameters are *shared* (not copied) from the source block.
    """

    def __init__(self, naive_block):
        super().__init__()
        _import_kernels_modernbert()

        # Share all parameters — no copy
        # Use getattr for optional bias attributes (MinimalModernBertSVDBlock may not have them)
        self.Pq = naive_block.Pq
        self.Vq = naive_block.Vq
        self.bq = getattr(naive_block, 'bq', None)
        self.Pk = naive_block.Pk
        self.Vk = naive_block.Vk
        self.bk = getattr(naive_block, 'bk', None)
        self.Pv = naive_block.Pv
        self.Vv = naive_block.Vv
        self.bv = getattr(naive_block, 'bv', None)

        self.Uo = naive_block.Uo
        self.Vo = naive_block.Vo
        self.bo_attn = getattr(naive_block, 'bo_attn', None)

        self.U1 = naive_block.U1
        self.V1 = naive_block.V1
        self.b1 = getattr(naive_block, 'b1', None)
        self.U2 = naive_block.U2
        self.V2 = naive_block.V2
        self.b2 = getattr(naive_block, 'b2', None)

        self.attn_norm = naive_block.attn_norm
        self.mlp_norm  = naive_block.mlp_norm
        self.rotary_emb = naive_block.rotary_emb

        self.num_heads   = naive_block.num_heads
        self.head_dim    = naive_block.head_dim
        self.hidden_size = getattr(naive_block, 'hidden_size', naive_block.num_heads * naive_block.head_dim)
        self.ffn_is_geglu     = naive_block.ffn_is_geglu
        self.gelu_approximate = naive_block.gelu_approximate
        self.attention_type   = getattr(naive_block, 'attention_type', 'global')

        self._rope_attn = None  # unused — SDPA is called directly with the HF mask

    def forward(self, hidden_states, attention_mask=None, sliding_window_mask=None,
                position_ids=None, output_attentions=False, **kwargs):
        from src.encoders.blocks_misc import _apply_rotary

        B, M, D = hidden_states.shape
        H, dh = self.num_heads, self.head_dim
        x = hidden_states

        # === Attention (unchanged from NaiveModernBertSVDBlock) ===
        xn = self.attn_norm(x)

        def project(xn, P, V, b):
            tmp = torch.einsum("bmd,hdr->bhmr", xn, P)
            out = torch.einsum("bhmr,hrd->bhmd", tmp, V)
            if b is not None:
                out = out + b.view(1, H, 1, dh)
            return out

        Q = project(xn, self.Pq, self.Vq, self.bq)
        K = project(xn, self.Pk, self.Vk, self.bk)
        V = project(xn, self.Pv, self.Vv, self.bv)

        if position_ids is None:
            position_ids = torch.arange(M, device=x.device).unsqueeze(0).expand(B, M)
        qf = Q.reshape(B * H, M, dh)
        kf = K.reshape(B * H, M, dh)
        posf = position_ids.unsqueeze(1).expand(B, H, M).reshape(B * H, M)
        try:
            cos, sin = self.rotary_emb(qf, position_ids=posf, layer_type=getattr(self, 'attention_type', 'global'))
        except (TypeError, KeyError):
            cos, sin = self.rotary_emb(qf, position_ids=posf)
        Q = _apply_rotary(qf, cos, sin).view(B, H, M, dh)
        K = _apply_rotary(kf, cos, sin).view(B, H, M, dh)

        # SDPA: mirror NaiveModernBertSVDBlock exactly — pass mask directly to PyTorch SDPA.
        # PyTorch dispatches to Flash Attention 2 which handles local windows natively in one
        # kernel launch.  The previous chunked-Python-loop path fired multiple SDPA calls and
        # caused a ~2× slowdown for ModernBERT local-attention layers.
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
        attn = F.scaled_dot_product_attention(Q, K, V, attn_mask=sdpa_mask, dropout_p=0.0)
        attn = attn.transpose(1, 2).reshape(B, M, D)
        attn_out = (attn @ self.Uo) @ self.Vo
        if self.bo_attn is not None:
            attn_out = attn_out + self.bo_attn
        x = x + attn_out

        # === FFN: fused GeGLU Triton kernel ===
        xn2 = self.mlp_norm(x)
        P_ffn = xn2 @ self.U1          # [B, M, R1] — rank-space input

        # GeGLU kernel requires non-None biases; substitute zeros if absent
        b1 = self.b1 if self.b1 is not None else torch.zeros(
            self.V1.shape[1], device=x.device, dtype=x.dtype)
        b2 = self.b2 if self.b2 is not None else torch.zeros(
            self.V2.shape[1], device=x.device, dtype=x.dtype)

        if self.ffn_is_geglu:
            y = _flashsvd_ffn_geglu_v15_fn(
                P_ffn, self.V1, self.U2, self.V2, b1, b2,
                gelu_approx=self.gelu_approximate,
            )
        else:
            # Non-GeGLU fallback (plain GELU, no split): stay in PyTorch
            z = P_ffn @ self.V1
            if self.b1 is not None:
                z = z + self.b1
            h = F.gelu(z, approximate=self.gelu_approximate)
            y = (h @ self.U2) @ self.V2
            if self.b2 is not None:
                y = y + self.b2

        x = x + y

        if output_attentions:
            return (x, None)
        return x


# ---------------------------------------------------------------------------
# Public API – v1.5
# ---------------------------------------------------------------------------
def enable_sdpa(model: nn.Module) -> nn.Module:
    """Switch every NaiveSVDBlock to use F.scaled_dot_product_attention.

    The rank-space projections (P/V matrices) remain as PyTorch einsum ops;
    only the attention score+softmax+weighted-sum step is replaced with
    PyTorch's fused SDPA (Flash Attention 2 / memory-efficient, if available).

    This provides an ablation point between:
      naive     — explicit einsum, full [B,H,M,M] matrix in HBM
      sdpa      — fused Flash Attention, no M² materialization, still PyTorch
      flashsvd  — Triton-fused rank-projection + attention + lift
      flashsvd15— Triton-fused rank-space kernel, native bf16/fp16
    """
    model_type = getattr(model.config, "model_type", "").lower()
    if model_type == "modernbert":
        # ModernBERT NaiveModernBertSVDBlock already uses SDPA unconditionally.
        print("[sdpa] ModernBERT already uses SDPA — no-op.")
        return model

    if hasattr(model, "bert"):
        encoder_layers = model.bert.encoder.layer
    elif hasattr(model, "roberta"):
        encoder_layers = model.roberta.encoder.layer
    else:
        raise RuntimeError(
            "enable_sdpa: cannot find .bert.encoder.layer or "
            ".roberta.encoder.layer on the supplied model."
        )

    patched = 0
    for layer in encoder_layers:
        block = getattr(layer, "block", None)
        if block is not None and hasattr(block, "attn_mode"):
            block.attn_mode = "sdpa"
            patched += 1

    if patched == 0:
        raise RuntimeError(
            "enable_sdpa: no NaiveSVDBlock instances found "
            "(FlashSVD blocks don't have attn_mode). "
            "Use --backend sdpa only before enable_flashsvd/flashsvd15."
        )
    print(f"[sdpa] Switched {patched} encoder layers to SDPA attention.")
    return model


def enable_flashsvd15(model: nn.Module) -> nn.Module:
    """Patch a model in-place: swap every NaiveSVDBlock for a FlashSVD15Block.

    Parameters
    ----------
    model : nn.Module
        A ``BertForSequenceClassification`` (or similar) whose encoder layers
        have been wrapped with ``BertLayerShim(NaiveSVDBlock(...))``.

    Returns
    -------
    model : nn.Module   (same object, mutated in-place)
    """
    _import_kernels_v15()  # fail-fast if kernels are missing

    # Locate encoder layers
    model_type = getattr(model.config, "model_type", "").lower()

    if model_type == "modernbert":
        _import_kernels_modernbert()
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            encoder_layers = model.model.layers
        else:
            raise RuntimeError(
                "enable_flashsvd15: cannot find .model.layers on ModernBERT model."
            )

        patched = 0
        already_flash = 0
        for layer in encoder_layers:
            block = getattr(layer, "block", None)
            _cls_name = type(block).__name__ if block is not None else ""

            if _cls_name == "FlashModernBertSVDBlock" or isinstance(block, FlashModernBertSVDBlock):
                already_flash += 1
                continue

            is_svd_block = (
                isinstance(block, NaiveModernBertSVDBlock)
                or (MinimalModernBertSVDBlock and isinstance(block, MinimalModernBertSVDBlock))
                or _cls_name in ("NaiveModernBertSVDBlock", "MinimalModernBertSVDBlock")
            )
            if is_svd_block:
                layer.block = FlashModernBertSVDBlock(block)
                patched += 1

        if patched == 0 and already_flash == 0:
            raise RuntimeError(
                "enable_flashsvd15: no NaiveModernBertSVDBlock instances found in ModernBERT model."
            )
        if already_flash > 0 and patched == 0:
            print(f"[flashsvd15] ModernBERT already enabled ({already_flash} layers) — no-op.")
        else:
            print(f"[flashsvd15] Patched {patched} ModernBERT layers with fused GeGLU kernel.")
        return model

    encoder_layers = None
    if hasattr(model, "bert"):
        encoder_layers = model.bert.encoder.layer
    elif hasattr(model, "roberta"):
        encoder_layers = model.roberta.encoder.layer
    else:
        raise RuntimeError(
            "enable_flashsvd15: cannot find .bert.encoder.layer or "
            ".roberta.encoder.layer on the supplied model."
        )

    patched = 0
    already_flash = 0
    for i, layer in enumerate(encoder_layers):
        block = getattr(layer, "block", None)
        _cls_name = type(block).__name__ if block is not None else ""

        # Idempotency: skip layers that are already FlashSVD15Block
        if _cls_name == "FlashSVD15Block" or isinstance(block, FlashSVD15Block):
            already_flash += 1
            continue

        is_svd_block = (
            isinstance(block, NaiveSVDBlock)
            or (MinimalSVDBlock and isinstance(block, MinimalSVDBlock))
            or _cls_name in ("NaiveSVDBlock", "MinimalSVDBlock", "FlashSVDBlock")
        )
        if is_svd_block:
            flash15_block = FlashSVD15Block(block)
            layer.block = flash15_block
            patched += 1

    if patched == 0 and already_flash == 0:
        raise RuntimeError(
            "enable_flashsvd15: no SVD block instances found. "
            "Did you run compression (--method != dense) before calling "
            "enable_flashsvd15?"
        )

    if already_flash > 0 and patched == 0:
        print(f"[flashsvd15] Already enabled ({already_flash} layers) — no-op.")
    else:
        print(f"[flashsvd15] Patched {patched} encoder layers with FlashSVD v1.5 kernels.")
    return model
