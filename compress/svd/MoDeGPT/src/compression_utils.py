import logging
import types
import math

import torch
from torch.types import Tensor


logger = logging.getLogger("MoDeGPT")

from src.model_utils import d1, d2, dtype_p

from torch.nn.functional import softmax


def bounded_rank_from_keep_ratio(
    dim: int,
    keep_ratio: float,
    *,
    min_rank: int = 1,
    even: bool = False,
) -> int:
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    ratio = float(keep_ratio)
    if not math.isfinite(ratio):
        raise ValueError(f"keep_ratio must be finite, got {keep_ratio}")

    ratio = min(1.0, max(0.0, ratio))
    min_rank = min(max(1, int(min_rank)), dim)
    rank = max(min_rank, min(int(dim * ratio), dim))

    if even and dim >= 2:
        rank -= rank % 2
        rank = max(2, rank)
        if rank > dim:
            rank = dim - (dim % 2)

    return rank


@torch.no_grad
def sqrt_M(
    M: Tensor, ridge_lambda=1e-4, scaled=False, debug: str = "", inverse_sqrt=False
) -> Tensor:
    # M_reg = M + torch.eye(M.shape[0], device=M.device, dtype=M.dtype) * ridge_lambda * scale

    eigenvalues, eigenvectors = torch.linalg.eigh(M)

    max_eig = eigenvalues.max()
    min_eig = eigenvalues.min()
    mean_eig = eigenvalues.mean().item()
    condition_ratio = max_eig / (min_eig + 1e-9)  # avoid div by zero

    if debug:
        print(f"{debug} Pre-reg: {max_eig:.1e} / {min_eig:.1e} = {condition_ratio:.1e}")
        print(f"{debug} Pre-reg: eigen.mean() = {mean_eig:.1e}")
    if min_eig < 0:
        print(f"Warning: Negative eigenvalues found ({min_eig}). Matrix is not PSD.")

    # scale = mean_eig if scaled else 1.0
    scale = max_eig if scaled else 1.0
    eigenvalues = eigenvalues + ridge_lambda * scale

    max_eig = eigenvalues.max()
    min_eig = eigenvalues.min()
    mean_eig = eigenvalues.mean().item()
    condition_ratio = max_eig / (min_eig + 1e-9)  # avoid div by zero

    if debug:
        print(f"{debug} Post-reg: {max_eig:.1e} / {min_eig:.1e} = {condition_ratio:.1e}")
        print(f"{debug} Post-reg eigen.mean() = {mean_eig:.1e}")

    sqrt_eigenvalues = torch.sqrt(eigenvalues.clamp(min=0))
    sqrt_M: Tensor = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T

    if not inverse_sqrt:
        return sqrt_M.to(dtype=M.dtype)
    else:
        inv_sqrt_eigenvalues = 1.0 / sqrt_eigenvalues.clamp(min=1e-12)
        inv_sqrt_M: Tensor = eigenvectors @ torch.diag(inv_sqrt_eigenvalues) @ eigenvectors.T
        return sqrt_M.to(dtype=M.dtype), inv_sqrt_M.to(dtype=M.dtype)


def get_gate_projs(model, layer_idx):
    try:
        block = model.model.decoder.layers[layer_idx]  # OPT
        up = block.fc1  # [D_int, D_h]
        down = block.fc2  # [D_h, D_int]
        return block, up, down, None, "opt"
    except AttributeError:
        try:
            block = model.transformer.h[layer_idx]  # GPT
            up = block.mlp.c_fc
            down = block.mlp.c_proj

            return block, up, down, None, "gpt"
        except AttributeError:
            block = model.model.layers[layer_idx]  # LLaMA
            up = block.mlp.up_proj
            down = block.mlp.down_proj
            gate = block.mlp.gate_proj
            return block, up, down, gate, "llama"


def allocate_global_sparsity(
    bi_scores: list[float],
    compression_ratio: float,
    # smoothing: float = 0.015,
    smoothing: float = 0.015,
    max_sparsity: float = 0.8,
    adapter=None,
    invert=False,
):
    if adapter:
        adapter.metrics["smoothing"] = smoothing

    n_layers = len(bi_scores)
    if n_layers == 0:
        raise ValueError("bi_scores must be non-empty")

    compression_ratio = float(compression_ratio)
    smoothing = float(smoothing)
    max_sparsity = float(max_sparsity)
    if not math.isfinite(compression_ratio) or not 0.0 <= compression_ratio <= 1.0:
        raise ValueError(f"compression_ratio must be finite and in [0, 1], got {compression_ratio}")
    if not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError(f"smoothing must be finite and positive, got {smoothing}")
    if not math.isfinite(max_sparsity) or not 0.0 <= max_sparsity <= 1.0:
        raise ValueError(f"max_sparsity must be finite and in [0, 1], got {max_sparsity}")

    epsilon = smoothing

    s = torch.tensor(bi_scores).to(dtype_p)
    if not torch.isfinite(s).all():
        raise ValueError("bi_scores contains NaN or Inf")
    if invert:
        s = -s

    # phi = L * phi_avg * softmax(-s / epsilon, dim=0)
    # the -s flips when using the CKA (higher score more compression)
    total_budget = n_layers * compression_ratio
    softmax_weights = softmax(-s / epsilon, dim=0)
    initial_sparsities = softmax_weights * total_budget

    logger.info(
        f"Max Layer Sparsity: {initial_sparsities.max().item()}, Avg = {initial_sparsities.mean().item()}"
    )
    if adapter:
        adapter.metrics["max_layer_sparsity"] = initial_sparsities.max().item()

    max_budget = n_layers * max_sparsity
    if total_budget > max_budget:
        logger.warning(
            f"Requested sparsity budget {total_budget} exceeds cap capacity {max_budget}; "
            "clamping to max capacity."
        )
        total_budget = max_budget

    sparsities = torch.zeros_like(softmax_weights)
    active = torch.ones_like(softmax_weights, dtype=torch.bool)
    remaining_budget = torch.as_tensor(total_budget, dtype=softmax_weights.dtype, device=softmax_weights.device)

    for _ in range(n_layers):
        if not bool(active.any().item()) or float(remaining_budget.item()) <= 0:
            break

        active_idx = active.nonzero(as_tuple=True)[0]
        active_weights = softmax_weights[active]
        active_weight_sum = active_weights.sum()
        if float(active_weight_sum.item()) <= 0:
            proposed = remaining_budget / active_idx.numel()
            proposed_sparsities = torch.full_like(active_weights, proposed)
        else:
            proposed_sparsities = remaining_budget * active_weights / active_weight_sum

        over_cap = proposed_sparsities > max_sparsity
        if not bool(over_cap.any().item()):
            sparsities[active_idx] = proposed_sparsities
            remaining_budget = remaining_budget - proposed_sparsities.sum()
            break

        capped_idx = active_idx[over_cap]
        sparsities[capped_idx] = max_sparsity
        remaining_budget = remaining_budget - (max_sparsity * capped_idx.numel())
        active[capped_idx] = False

    remaining = float(remaining_budget.item())
    if remaining > 1e-3:
        logger.warning(f"Unallocated sparsity budget after capping: {remaining}")

    sparsities = sparsities.clamp(min=0.0, max=max_sparsity)
    logger.info(
        f"Capped Layer Sparsity: {sparsities.max().item()}, Avg = {sparsities.mean().item()}"
    )

    keep_ratios = (1 - sparsities).clamp(min=0.0, max=1.0)
    return keep_ratios.tolist()
