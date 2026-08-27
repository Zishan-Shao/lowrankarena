"""
Unified loss computation for SVD-LLM compression.

Consolidates repeated CE loss patterns.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict

from compression.batch_utils import _model_forward


def compute_ce_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute cross-entropy loss with label shifting.

    Args:
        logits: Model output logits [B, T, V]
        input_ids: Input token IDs [B, T]
        attention_mask: Optional attention mask [B, T]
        token_mask: Optional per-token mask for selecting specific tokens [B, T-1]

    Returns:
        Scalar loss tensor
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    per_tok = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(input_ids.size(0), -1)  # [B, T-1]

    if attention_mask is not None:
        valid = attention_mask[..., 1:].contiguous().bool()
    else:
        valid = torch.ones_like(per_tok, dtype=torch.bool)

    if token_mask is not None:
        mask = valid & token_mask.bool()
        if mask.any():
            return per_tok[mask].mean()
        else:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
    else:
        return per_tok[valid].mean()


def compute_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Cross-entropy loss dispatcher (kept for backwards-compatible call sites).

    Args:
        logits: Model output logits [B, T, V]
        input_ids: Input token IDs [B, T]
        attention_mask: Optional attention mask [B, T]
        token_mask: Optional per-token mask

    Returns:
        Scalar loss tensor
    """
    return compute_ce_loss(logits, input_ids, attention_mask, token_mask)


def compute_per_sample_ce_loss(
    model,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    autocast_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Compute per-sample CE loss for a batch (used in hardest sample selection).

    Args:
        model: The model to evaluate
        batch: Batch dict with 'input_ids' and optionally 'attention_mask'
        device: Device to run on
        autocast_dtype: Autocast dtype for mixed precision

    Returns:
        Per-sample loss tensor [B]
    """
    ids = batch["input_ids"].to(device)
    am = batch.get("attention_mask", torch.ones_like(ids)).to(device)

    forward_batch = {"input_ids": ids, "attention_mask": am}

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=autocast_dtype):
        logits = _model_forward(model, forward_batch, device).logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = ids[..., 1:].contiguous()

    per_tok = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(ids.size(0), -1)  # [B, T-1]

    valid = am[..., 1:].contiguous().bool()

    # Mean per sample
    per_sample_loss = []
    for i in range(ids.size(0)):
        sample_valid = valid[i]
        if sample_valid.any():
            per_sample_loss.append(per_tok[i][sample_valid].mean())
        else:
            per_sample_loss.append(torch.tensor(0.0, device=device))

    return torch.stack(per_sample_loss)
