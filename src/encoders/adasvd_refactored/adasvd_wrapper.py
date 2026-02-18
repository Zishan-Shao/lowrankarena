"""
Wrapper module for integrating refactored AdaSVD into run_encoder_benchmark.py

Provides clean APIs for:
1. Training AdaSVD to generate ranks.json
2. Compressing models with naive backend
3. Compressing models with FlashSVD backend
"""

import os
import sys
import json
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from transformers import PreTrainedModel
from torch.utils.data import DataLoader

# Add current directory to path for local imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


def train_adasvd_ranks(
    model: PreTrainedModel,
    calib_loader: DataLoader,
    budget: float,
    output_dir: str = "ars_out",
    steps: int = 400,
    device: str = "cuda"
) -> Dict[str, int]:
    """
    Train AdaSVD hypernetwork to generate per-operation ranks.

    Args:
        model: HuggingFace model (e.g., BertForSequenceClassification)
        calib_loader: Calibration data loader
        budget: Target parameter ratio (e.g., 0.4 = 40% of original params)
        output_dir: Directory to save ranks.json and budget_report.json
        steps: Number of training steps
        device: Device to run on

    Returns:
        Dict mapping operation names to ranks
    """
    from adaptive_rank_selection import (
        set_seed, collect_linear_modules, replace_with_masked,
        SimpleHN, parameter_budget, alignment_loss
    )

    set_seed(42)
    model = model.to(device).eval()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all Linear modules and their capacities
    linear_list = collect_linear_modules(model)
    op_names = [n for n, _ in linear_list]
    R_caps = []
    for name, lin in linear_list:
        cap = min(lin.in_features, lin.out_features)
        R_caps.append(cap)

    # Replace Linear layers with MaskedSVDLinear
    print(f"Replacing {len(op_names)} Linear layers with MaskedSVDLinear...")
    model, masked_ops = replace_with_masked(model, device=device, rank_cap_per_op=R_caps, original_names=op_names)
    op_list = [op for _, op in masked_ops]

    # Create hypernetwork
    HN = SimpleHN(op_sizes=R_caps, feat_dim=16, hidden=64).to(device)
    optimizer = torch.optim.Adam(HN.parameters(), lr=1e-3)

    # Training loop
    print(f"Training AdaSVD hypernetwork for {steps} steps (budget={budget})...")
    batch_iter = iter(calib_loader)
    for step in range(steps):
        # Get batch
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(calib_loader)
            batch = next(batch_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Get masks from hypernetwork
        logits_list = HN()
        from adaptive_rank_selection import gumbel_sigmoid
        masks_soft = [gumbel_sigmoid(l, tau=1.0, hard=True) for l in logits_list]

        # Set masks for each operation
        for op, mask in zip(op_list, masks_soft):
            op._current_mask = mask.to(op.U.dtype)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        task_loss = outputs.loss

        # Budget loss
        budget_loss = parameter_budget(op_list, masks_soft, budget)

        # Alignment loss
        align_loss = sum(alignment_loss(m, op.s) for op, m in zip(op_list, masks_soft))

        # Total loss (FIX: Two-sided budget constraint with small alignment)
        # Budget loss is now squared error: ((Tm-Tmax)/Tmax)^2
        # For target=0.3, if ratio=0.9: budget_loss = ((0.9-0.3)/0.3)^2 = 4.0
        # lambda_param=100.0 (scaled for squared error), gamma_align=0.01 (minimal)
        loss = task_loss + 100.0 * budget_loss + 0.01 * align_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            with torch.no_grad():
                total_params = sum((op.in_features + op.out_features) * m.sum().item()
                                   for op, m in zip(op_list, masks_soft))
                # FIX: Compute ratio relative to ORIGINAL model params, not SVD params
                original_params = sum(op.in_features * op.out_features for op in op_list)
                ratio = total_params / original_params
                print(f"  Step {step:3d} | Loss: {loss.item():.4f} | Task: {task_loss.item():.4f} | Budget: {budget_loss.item():.4f} | Align: {align_loss.item():.4f} | Ratio: {ratio:.3f} | Target: {budget:.3f}")

    # Finalize masks and extract ranks
    print("Finalizing masks and extracting ranks...")
    with torch.no_grad():
        logits_list = HN()
        masks_soft = [torch.sigmoid(l) for l in logits_list]
        ks = [int(torch.clamp(torch.round(m.sum()), 0, m.numel()).item()) for m in masks_soft]

        # Convert to binary top-k masks
        final_masks = []
        for m_soft, k in zip(masks_soft, ks):
            m_binary = torch.zeros_like(m_soft)
            if k > 0:
                m_binary[:k] = 1.0
            final_masks.append(m_binary)

        # Set final masks
        for op, m_final in zip(op_list, final_masks):
            op._current_mask = m_final

    # Build ranks dictionary
    ranks_dict = {name: k for name, k in zip(op_names, ks)}

    # Calculate achieved ratio (FIX: Relative to ORIGINAL model params)
    total_params = sum((op.in_features + op.out_features) * k
                       for op, k in zip(op_list, ks))
    original_params = sum(op.in_features * op.out_features for op in op_list)
    achieved_ratio = total_params / original_params

    # Save ranks.json
    ranks_path = os.path.join(output_dir, "ranks.json")
    with open(ranks_path, "w") as f:
        json.dump(ranks_dict, f, indent=2)
    print(f"Saved ranks to: {ranks_path}")

    # Save budget report
    budget_report = {
        "target_budget": budget,
        "achieved_ratio": achieved_ratio,
        "total_params": int(total_params),
        "original_model_params": int(original_params),
        "num_operations": len(op_list)
    }
    report_path = os.path.join(output_dir, "budget_report.json")
    with open(report_path, "w") as f:
        json.dump(budget_report, f, indent=2)
    print(f"Saved budget report to: {report_path}")
    print(f"  Target: {budget:.3f} | Achieved: {achieved_ratio:.3f}")

    return ranks_dict


def compress_adasvd_naive(
    model: PreTrainedModel,
    ranks_path: str,
    device: str = "cuda"
) -> PreTrainedModel:
    """
    Compress model with AdaSVD using naive backend (standard PyTorch).

    Args:
        model: HuggingFace model
        ranks_path: Path to ranks.json file
        device: Device to run on

    Returns:
        Compressed model
    """
    from profile_svd import (
        attach_fullnames, build_plain_svd_helpers,
        FWSVDBlock, LayerShim
    )

    model = model.to(device).eval()

    # Load ranks
    with open(ranks_path, "r") as f:
        ranks_dict = json.load(f)

    print(f"Loaded {len(ranks_dict)} per-operation ranks from {ranks_path}")

    # Attach names for rank lookup
    attach_fullnames(model)

    # Build SVD helpers
    svd_per_head, svd_low_rank = build_plain_svd_helpers(model)

    # Get encoder layers
    if hasattr(model, 'bert'):
        encoder_layers = model.bert.encoder.layer
    elif hasattr(model, 'roberta'):
        encoder_layers = model.roberta.encoder.layer
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    # Replace each layer with FWSVDBlock
    print(f"Replacing {len(encoder_layers)} layers with FWSVDBlock (naive backend)...")
    for i, layer in enumerate(encoder_layers):
        block = FWSVDBlock(layer, ranks_dict, svd_per_head, svd_low_rank)
        shimmed = LayerShim(block)
        encoder_layers[i] = shimmed.to(device).eval()

    # Cleanup
    del svd_per_head, svd_low_rank
    torch.cuda.empty_cache()

    return model


def compress_adasvd_flashsvd(
    model: PreTrainedModel,
    ranks_path: str,
    ffn_kernel: str = "v1",
    device: str = "cuda"
) -> PreTrainedModel:
    """
    Compress model with AdaSVD using FlashSVD backend (Triton kernels).

    Args:
        model: HuggingFace model
        ranks_path: Path to ranks.json file
        ffn_kernel: FFN kernel variant ("v1" or "v2")
        device: Device to run on

    Returns:
        Compressed model
    """
    import numpy as np
    from profile_flashsvd import (
        attach_fullnames, build_plain_svd_helpers,
        FlashSVDBlock, LayerShim
    )

    model = model.to(device).eval()

    # Load ranks
    with open(ranks_path, "r") as f:
        ranks_dict = json.load(f)

    print(f"Loaded {len(ranks_dict)} per-operation ranks from {ranks_path}")

    # Check budget and use median rank for low budgets (<0.3)
    ranks_dir = os.path.dirname(ranks_path)
    budget_report_path = os.path.join(ranks_dir, "budget_report.json")

    if os.path.exists(budget_report_path):
        with open(budget_report_path, "r") as f:
            budget_report = json.load(f)
        target_budget = budget_report.get("target_budget", 1.0)

        # Use median rank strategy for low budgets to ensure FlashSVD compatibility
        if target_budget < 0.3:
            ranks_list = [r for r in ranks_dict.values() if r > 0]
            median_rank = int(np.median(ranks_list))
            original_ranks = ranks_dict.copy()

            # Replace all ranks with median
            ranks_dict = {k: median_rank for k in ranks_dict.keys()}

            print(f"[FlashSVD Low-Budget Strategy] budget={target_budget:.1f} < 0.3")
            print(f"  Using median rank={median_rank} uniformly (FlashSVD compatibility)")
            print(f"  Original ranks: min={min(ranks_list)}, max={max(ranks_list)}, range={max(ranks_list)-min(ranks_list)}")
        else:
            print(f"[FlashSVD Per-Op Strategy] budget={target_budget:.1f} >= 0.3, using adaptive ranks")

    # Attach names for rank lookup
    attach_fullnames(model)

    # Build SVD helpers
    svd_per_head, svd_low_rank = build_plain_svd_helpers(model)

    # Get encoder layers
    if hasattr(model, 'bert'):
        encoder_layers = model.bert.encoder.layer
    elif hasattr(model, 'roberta'):
        encoder_layers = model.roberta.encoder.layer
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    # Replace each layer with FlashSVDBlock
    print(f"Replacing {len(encoder_layers)} layers with FlashSVDBlock (ffn_kernel={ffn_kernel})...")
    for i, layer in enumerate(encoder_layers):
        block = FlashSVDBlock(layer, ranks_dict, svd_per_head, svd_low_rank, ffn_kernel=ffn_kernel)
        shimmed = LayerShim(block)
        encoder_layers[i] = shimmed.to(device).eval()

    # Cleanup
    del svd_per_head, svd_low_rank
    torch.cuda.empty_cache()

    return model


def get_param_ratio_from_budget_report(output_dir: str = "ars_out") -> float:
    """
    Get achieved parameter ratio from budget_report.json.

    Args:
        output_dir: Directory containing budget_report.json

    Returns:
        Achieved parameter ratio
    """
    report_path = os.path.join(output_dir, "budget_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = json.load(f)
            return report.get("achieved_ratio", 0.5)
    return 0.5  # Default fallback
