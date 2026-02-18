"""
FlashSVD backend: replaces NaiveSVDBlock forward passes with Triton-accelerated
flash_svd_attention + flashsvd_ffn kernels.

Usage:
    from eval_encoder.flashsvd_backend import enable_flashsvd
    enable_flashsvd(model)          # in-place, raises if kernels unavailable
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_encoder.blocks import NaiveSVDBlock, BertLayerShim

# Import MinimalSVDBlock if available (for models loaded from checkpoint)
try:
    from eval_encoder.load_compressed_model import MinimalSVDBlock
except ImportError:
    MinimalSVDBlock = None

# ---------------------------------------------------------------------------
# Lazy kernel imports – fail fast with a human-readable message
# ---------------------------------------------------------------------------
_flash_svd_attention = None
_flashsvd_ffn_v1 = None


def _import_kernels():
    global _flash_svd_attention, _flashsvd_ffn_v1
    if _flash_svd_attention is not None:
        return  # already imported

    # Ensure the encoder-kernel directory is importable
    # Try local kernels first (for standalone/Docker deployment)
    local_kernel_dir = os.path.join(os.path.dirname(__file__), "kernels")
    # Fall back to repository structure
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_kernel_dir = os.path.join(_REPO_ROOT, "kernels", "encoder_kernels")

    if os.path.isdir(local_kernel_dir):
        kernel_dir = local_kernel_dir
    elif os.path.isdir(repo_kernel_dir):
        kernel_dir = repo_kernel_dir
    else:
        raise RuntimeError(
            f"FlashSVD kernel directory not found.\n"
            f"Tried: {local_kernel_dir} and {repo_kernel_dir}\n"
            "Make sure kernels are available."
        )

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
        # Pq stored as [1, H, dm, R]
        Pq = self.Pq[0]  # [H, dm, R]
        H = Pq.shape[0]
        R = Pq.shape[2]
        dh = dm // H

        # --- project x into low-rank space per head ---
        tmp_q = torch.einsum("bmd,hdr->bhmr", x, Pq).contiguous()
        tmp_k = torch.einsum("bmd,hdr->bhmr", x, self.Pk[0]).contiguous()
        tmp_v = torch.einsum("bmd,hdr->bhmr", x, self.Pv[0]).contiguous()

        # expand V / bias to [B, H, ...] for the Triton kernel
        Vq_f = self.Vq[0].expand(B, H, R, dh)
        Vk_f = self.Vk[0].expand(B, H, R, dh)
        Vv_f = self.Vv[0].expand(B, H, R, dh)
        bq_f = self._bq_sq.expand(B, H, dh)
        bk_f = self._bk_sq.expand(B, H, dh)
        bv_f = self._bv_sq.expand(B, H, dh)

        # --- flash SVD attention ---
        mask4 = mask.view(B, 1, 1, M) if mask is not None else None
        attn_out = _flash_svd_attention(
            tmp_q, Vq_f, bq_f,
            tmp_k, Vk_f, bk_f,
            tmp_v, Vv_f, bv_f,
            mask=mask4,
            block_m=32,
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
            "enable_flashsvd: ModernBERT requires different Triton kernels "
            "(flashsvdropeattn + flashsvdgeglu) that are not yet integrated "
            "in the benchmark pipeline. Use --backend naive for ModernBERT."
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
    for i, layer in enumerate(encoder_layers):
        block = getattr(layer, "block", None)
        # Support both NaiveSVDBlock (from compression) and MinimalSVDBlock (from checkpoint loading)
        is_svd_block = isinstance(block, NaiveSVDBlock) or (MinimalSVDBlock and isinstance(block, MinimalSVDBlock))
        if is_svd_block:
            flash_block = FlashSVDBlock(block)
            layer.block = flash_block
            patched += 1

    if patched == 0:
        raise RuntimeError(
            "enable_flashsvd: no NaiveSVDBlock or MinimalSVDBlock instances found. "
            "Did you run compression (--method != dense) before calling "
            "enable_flashsvd?"
        )

    print(f"[flashsvd] Patched {patched} encoder layers with FlashSVD kernels.")
    return model
