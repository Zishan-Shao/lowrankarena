import logging
import types

import torch
from torch.types import Tensor


logger = logging.getLogger("MoDeGPT")

from src.model_utils import d1, d2, dtype_p

from torch.nn.functional import softmax


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

    L = n_layers
    epsilon = smoothing

    s = torch.tensor(bi_scores).to(dtype_p)
    if invert:
        s = -s

    # phi = L * phi_avg * softmax(-s / epsilon, dim=0)
    # the -s flips when using the CKA (higher score more compression)
    total_budget = n_layers * compression_ratio
    softmax_weights = softmax(-s / epsilon, dim=0)
    sparsities = softmax_weights * total_budget

    logger.info(f"Max Layer Sparsity: {sparsities.max().item()}, Avg = {sparsities.mean().item()}")
    if adapter:
        adapter.metrics["max_layer_sparsity"] = sparsities.max().item()

    while True:
        clamped = sparsities > max_sparsity

        if not clamped.any():
            break

        excess = (sparsities[clamped] - max_sparsity).sum()
        sparsities[clamped] = max_sparsity

        free = ~clamped
        if free.any():
            # Redistribute proportionally among the non-capped experts
            sparsities[free] += excess * (softmax_weights[free] / softmax_weights[free].sum())

    return (1 - sparsities).tolist()
