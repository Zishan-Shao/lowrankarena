import logging

from torch import Tensor
import torch
from src.model_utils import dtype_p, d2
from src.adapters.model_adapter import MLPComponents, ModelAdapter

logger = logging.getLogger("MoDeGPT")

decay_scores = {}


def get_ridge_scores(C, layer_idx: int, ridge_lambda=1e-2) -> Tensor:

    C = C.to(dtype=dtype_p, device=d2)  # [D_int, D_int]
    D_int = C.shape[0]

    C_ridge = C + (ridge_lambda * torch.eye(D_int, device=C.device))

    L = torch.linalg.cholesky(C_ridge)
    C_ridge_inv = torch.cholesky_inverse(L)

    inv_diag = torch.diag(C_ridge_inv)

    return inv_diag


@torch.no_grad()
def compress_weights(
    comps: MLPComponents, C: Tensor, keep_ratio: float, layer_idx: int, ridge_lambda: float
):

    global decay_scores

    ridge_scores = get_ridge_scores(C, layer_idx=layer_idx, ridge_lambda=ridge_lambda)

    rank = int(C.shape[0] * keep_ratio)

    W_u = comps.up_proj.weight.to(device=d2, dtype=dtype_p)  # [D_int, D_h]
    W_g = comps.gate_proj.weight.to(device=d2, dtype=dtype_p)
    W_d = comps.down_proj.weight.to(device=d2, dtype=dtype_p)  # [D_h, D_int]

    inv_diag = ridge_scores

    topk = torch.topk(inv_diag, k=rank, largest=False, dim=0).indices

    topk, _ = torch.sort(topk)

    W_u_proj = W_u[topk, :].T
    W_g_proj = W_g[topk, :].T

    C_reduced = C[topk][:, topk]

    down_cross_term = C[topk, :] @ W_d.T  # [D_ff, d_h/r]

    L_red = torch.linalg.cholesky(C_reduced + (1e-6 * torch.eye(rank, device=d2, dtype=dtype_p)))
    W_d_proj = torch.cholesky_solve(down_cross_term, L_red)

    return (
        W_u_proj.to(dtype=torch.bfloat16),
        W_d_proj.to(dtype=torch.bfloat16),
        W_g_proj.to(dtype=torch.bfloat16),
        rank,
    )


@torch.no_grad()
def compress_nystrom(
    adapter: ModelAdapter,
    cov,
    keep_ratios,
    target_layers,
    ridge_lambda=1e-4,
):
    """
    Compresses d_ff dimension
    same as compress_mlp_moe except rank varies across experts
    """

    arch = adapter.arch

    for layer_idx in target_layers:
        weight_cache = {}

        comps = adapter.get_mlp_components(layer_idx)
        c = cov[layer_idx]

        W_u_proj, W_d_proj, W_g_proj, rank = compress_weights(
            comps,
            c,
            keep_ratios[layer_idx],
            layer_idx=layer_idx,
            ridge_lambda=adapter.config.nystrom_ridge,
        )
        logger.info(f"[MLP] ✅ Layer {layer_idx}  compressed to rank {rank}")

        weight_cache = {"up": W_u_proj.T, "gate": W_g_proj.T, "down": W_d_proj.T}

        adapter.save_layer(
            output_dir=adapter.config.temp_storage_dir,
            suffix="mlp",
            weights=weight_cache,
            layer_idx=layer_idx,
        )

        # adapter.slice_gate_dims(
        #     layer_idx=layer_idx,
        #     up_weights=W_u_proj.T,
        #     down_weights=W_d_proj.T,
        #     gate_weights=W_g_proj.T,
        #     new_bias_u=None,
        #     new_bias_g=None,
        #     bias=False,
        #     expert_idx=None,
        # )

        torch.cuda.empty_cache()
