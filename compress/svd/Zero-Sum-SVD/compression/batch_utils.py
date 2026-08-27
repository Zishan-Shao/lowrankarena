"""
Batch processing utilities for SVD-LLM compression.

Extracted from SVDLLM_sum0strategy_alllowrank_fullprune_eachstage_onescalaer_correction.py
Consolidates repeated batch processing patterns.
"""

import torch
from typing import Optional, Tuple, Dict, Any


def prepare_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    max_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Prepare a batch for model forward pass.

    Moves tensors to device, creates attention mask if missing,
    and optionally truncates sequences.

    Args:
        batch: Batch dict with 'input_ids' and optionally 'attention_mask', 'position_ids'
        device: Target device
        max_len: If provided, truncate sequences to this length (from the right)

    Returns:
        Tuple of (input_ids, attention_mask, position_ids)
    """
    ids = batch["input_ids"].to(device)
    am = batch.get("attention_mask", None)
    if am is not None:
        am = am.to(device)
    else:
        am = torch.ones_like(ids)

    position_ids = batch.get("position_ids", None)
    if position_ids is not None:
        position_ids = position_ids.to(device)

    # Truncate if needed
    if max_len is not None and ids.size(1) > max_len:
        ids = ids[:, -max_len:]
        am = am[:, -max_len:]
        if position_ids is not None:
            position_ids = position_ids[:, -max_len:]

    return ids, am, position_ids


def cleanup_gpu(device: torch.device) -> None:
    """Conditional GPU cache cleanup."""
    if device.type == "cuda":
        torch.cuda.empty_cache()


def build_model_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build model input dict from tensors.

    Args:
        input_ids: Input token IDs
        attention_mask: Attention mask
        position_ids: Optional position IDs

    Returns:
        Dict suitable for model(**inputs)
    """
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if position_ids is not None:
        inputs["position_ids"] = position_ids
    return inputs


def get_batch_size(batch: Dict[str, torch.Tensor]) -> int:
    """Get batch size from batch dict."""
    return batch["input_ids"].size(0)


def get_seq_len(batch: Dict[str, torch.Tensor]) -> int:
    """Get sequence length from batch dict."""
    return batch["input_ids"].size(1)


def _model_forward(model, batch: Dict[str, torch.Tensor], dev: torch.device):
    """
    Run a forward pass on the model with text inputs from the batch.

    Args:
        model: The model to run forward pass on
        batch: Batch dict with 'input_ids', 'attention_mask', and optional 'position_ids'
        dev: Device to move tensors to

    Returns:
        Model output (typically has .logits attribute)
    """
    inputs = {}
    for k in ["input_ids", "attention_mask", "position_ids"]:
        if k in batch:
            inputs[k] = batch[k].to(dev)
    return model(**inputs)
