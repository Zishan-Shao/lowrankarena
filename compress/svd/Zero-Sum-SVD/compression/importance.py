"""
Importance computation for SVD-LLM compression.

Core gradient and importance functions extracted from
SVDLLM_sum0strategy_alllowrank_fullprune_eachstage_onescalaer_correction.py

Note: The main recompute_importance() function remains in the main file
due to its many dependencies. This module contains the helper functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

from utils.model_utils import find_target_modules_in_layer
from compression.batch_utils import _model_forward


def _importance_autocast_dtype(model) -> torch.dtype:
    """Determine autocast dtype based on model parameter types."""
    return torch.bfloat16 if any(p.dtype == torch.bfloat16 for p in model.parameters()) else torch.float16


def right_solve_Rt(G: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Solve G @ L^{-T} using triangular solve.
    Returns H = G @ L^{-T}.
    """
    # G @ L^{-T} = (L^{-1} @ G^T)^T
    return torch.linalg.solve_triangular(L, G.transpose(0, 1), upper=False).transpose(0, 1)


@torch.no_grad()
def compute_importance_for_layer_fast(
    W: torch.Tensor,
    G: torch.Tensor,
    L: torch.Tensor,
    use_triangular_solve: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute singular value importance for a single layer.

    Args:
        W: Weight matrix [m, n]
        G: Average gradient [m, n]
        L: Whitening matrix [n, n]
        use_triangular_solve: Use triangular solve instead of explicit inverse

    Returns:
        Tuple of (S, grad_sigma, saliency, U, Vh)
    """
    A = W @ L
    if use_triangular_solve:
        H = right_solve_Rt(G, L)
    else:
        try:
            L_inv = torch.linalg.inv(L)
        except RuntimeError:
            L = L + 1e-6 * torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
            L_inv = torch.linalg.inv(L)
        H = G @ L_inv.transpose(0, 1)

    eps = 1e-12

    # Use standard SVD: A = U diag(S) V^T
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    # Sort singular values in descending order
    idx = torch.argsort(S, descending=True)
    S = S[idx]
    U = U[:, idx]
    Vh = Vh[idx, :]

    V = Vh.transpose(0, 1)

    # M gives grad wrt singular values
    M = U.transpose(0, 1) @ H @ V
    grad_sigma = torch.diagonal(M, dim1=-2, dim2=-1).contiguous()
    saliency = grad_sigma.abs()

    return S, grad_sigma, saliency, U, Vh


def compute_grad_sigma_efficient_for_module(
    model,
    cali_data: List[Dict[str, torch.Tensor]],
    dev: torch.device,
    mod: nn.Linear,
    U: torch.Tensor,
    V: torch.Tensor,
    L: torch.Tensor,
    args,
    autocast_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Computes grad_sigma for one module without storing full m x n gradients on CPU.
    Accumulates only diag(U^T @ H @ V) as a length r vector on CPU.
    """
    max_grad_samples = getattr(args, "nsamples_gradient_subset", None)
    if max_grad_samples is not None:
        try:
            max_grad_samples = int(max_grad_samples)
        except Exception:
            max_grad_samples = None
        if max_grad_samples is not None and max_grad_samples <= 0:
            max_grad_samples = None

    global_sample_ptr = 0
    batch_count = 0
    r = U.size(1)
    diag_sum_cpu = torch.zeros(r, device="cpu", dtype=torch.float32)

    total_samples = len(cali_data)
    if max_grad_samples is not None:
        total_samples = min(total_samples, max_grad_samples)

    for batch in tqdm(cali_data, desc="Efficient grad_sigma", total=total_samples, leave=False):
        if max_grad_samples is not None and global_sample_ptr >= max_grad_samples:
            break

        raw_ids = batch["input_ids"].to(dev)
        raw_am = batch.get("attention_mask", None)
        if raw_am is not None:
            raw_am = raw_am.to(dev)
        else:
            raw_am = torch.ones_like(raw_ids)

        raw_pi = batch.get("position_ids", None)
        if raw_pi is not None:
            raw_pi = raw_pi.to(dev)

        B_all = raw_ids.size(0)
        if max_grad_samples is not None:
            remaining = max_grad_samples - global_sample_ptr
            if remaining <= 0:
                break
            B_use = min(B_all, remaining)
        else:
            B_use = B_all

        if B_use <= 0:
            break

        ids = raw_ids[:B_use]
        am = raw_am[:B_use]
        position_ids = raw_pi[:B_use] if raw_pi is not None else None
        B = ids.size(0)

        # Trim to the maximum number of non-pad tokens in this batch
        actual_len = int(am.sum(dim=1).max().item())
        actual_len = max(actual_len, 2)  # need >=2 tokens for shift
        ids = ids[:, :actual_len]
        am = am[:, :actual_len]
        if position_ids is not None:
            position_ids = position_ids[:, :actual_len]

        max_len = None
        if getattr(args, "_in_importance_pass", False):
            max_len = getattr(args, "importance_seq_len", None)
        if max_len is not None and max_len > 0 and ids.size(1) > max_len:
            ids = ids[:, -max_len:]
            am = am[:, -max_len:]
            if position_ids is not None:
                position_ids = position_ids[:, -max_len:]

        forward_batch = {"input_ids": ids, "attention_mask": am}
        if position_ids is not None:
            forward_batch["position_ids"] = position_ids

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            out = _model_forward(model, forward_batch, dev)
            logits = out.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = ids[..., 1:].contiguous()

            per_tok = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(ids.size(0), -1)

            valid = am[..., 1:].contiguous().bool()

            tok_mask = batch.get("token_mask", None)
            if tok_mask is not None:
                tok_mask = tok_mask.to(dev).bool()
                mask = valid & tok_mask
                if mask.any():
                    loss = per_tok[mask].mean()
                else:
                    del out, logits, per_tok
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    global_sample_ptr += B
                    continue
            else:
                loss = per_tok[valid].mean()

        grad_tensor = torch.autograd.grad(
            loss,
            mod.weight,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]

        if grad_tensor is not None:
            G = grad_tensor.detach().to(dev, dtype=torch.float32)

            if getattr(args, "use_triangular_solve", False):
                H = right_solve_Rt(G, L)
            else:
                try:
                    L_inv = torch.linalg.inv(L)
                except RuntimeError:
                    L_reg = L + 1e-6 * torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
                    L_inv = torch.linalg.inv(L_reg)
                H = G @ L_inv.transpose(0, 1)

            UtH = U.transpose(0, 1) @ H
            diag = (UtH * V.transpose(0, 1)).sum(dim=1)
            diag_sum_cpu.add_(diag.detach().to("cpu", dtype=torch.float32))
            batch_count += 1

            del G, H, UtH, diag

        global_sample_ptr += B

        del out, logits, loss
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    n = max(batch_count, 1)
    grad_sigma = diag_sum_cpu.div_(float(n))
    return grad_sigma
