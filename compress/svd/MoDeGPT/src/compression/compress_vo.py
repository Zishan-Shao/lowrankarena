import logging

import torch
from torch.types import Tensor

from src.compression_utils import sqrt_M
from src.model_utils import dtype_p, d1, d2
from src.adapters.model_adapter import ModelAdapter

logger = logging.getLogger("MoDeGPT")


@torch.no_grad()
def compress_vo(
    adapter: ModelAdapter,
    cov: list[Tensor],
    keep_ratios=None,
    slice_dims=True,
    target_layers: list[int] | None = None,
):

    if target_layers is None:
        target_layers = [i for i in range(adapter.n_layers)]

    model = adapter.model
    n_layers = adapter.n_layers
    n_heads = adapter.n_heads
    head_dim = adapter.head_dim
    arch = adapter.arch
    n_kv_heads = adapter.n_kv_heads

    grouped = n_kv_heads != n_heads

    for layer in target_layers:
        keep_ratio = keep_ratios[layer]
        rank_i = int(head_dim * keep_ratio)
        rank_i = max(1, rank_i)

        if arch == "llama" or "qwen" in arch:
            rank_i = rank_i - (rank_i % 2)
            rank_i = max(2, rank_i)

        C = cov[layer].to(device=d2)
        sqrt_C = sqrt_M(C, ridge_lambda=adapter.config.ridge_vo, debug="Type 3 VO cov:")
        inv_sqrt_C = torch.linalg.inv(sqrt_C)

        try:
            comps = adapter.get_attn_components(layer)
            W_v = comps.v_proj.weight
            W_o = comps.o_proj.weight
        except Exception as e:
            logger.warning(f"[VO] Layer {layer}: cannot access v_proj/o_proj: {e}")
            continue

        new_heads_V = []
        new_heads_O = []

        for h in range(n_kv_heads):
            if grouped:
                compress_head_grouped(
                    kv_head_idx=h,
                    kv_head_ratio=n_heads // n_kv_heads,
                    head_dim=head_dim,
                    rank=rank_i,
                    W_v=W_v,
                    W_o=W_o,
                    sqrt_C=sqrt_C,
                    inv_sqrt_C=inv_sqrt_C,
                    new_heads_V=new_heads_V,
                    new_heads_O=new_heads_O,
                    slice_dims=slice_dims,
                    arch=arch,
                )
            else:
                compress_head(
                    head_idx=h,
                    head_dim=head_dim,
                    rank_i=rank_i,
                    W_v=W_v,
                    W_o=W_o,
                    sqrt_C=sqrt_C,
                    inv_sqrt_C=inv_sqrt_C,
                    new_heads_V=new_heads_V,
                    new_heads_O=new_heads_O,
                    slice_dims=slice_dims,
                    arch=arch,
                )

        V_heads = torch.cat(new_heads_V, dim=0).to(device="cuda", dtype=torch.bfloat16)
        O_heads = torch.cat(new_heads_O, dim=1).to(device="cuda", dtype=torch.bfloat16)

        weight_cache = {"v_proj": V_heads, "o_proj": O_heads}

        adapter.save_layer(
            output_dir=adapter.config.temp_storage_dir,
            suffix="vo",
            weights=weight_cache,
            layer_idx=layer,
        )
        # if slice_dims:
        #     adapter.slice_vo_dims(
        #         layer_idx=layer,
        #         new_heads_V=new_heads_V,
        #         new_heads_O=new_heads_O,
        #         bias=True if arch == "opt" else False,
        #     )

        if logger:
            logger.info(f"[VO] ✅ Compressed layer {layer} to rank {rank_i} per head")


@torch.no_grad()
def compress_head_grouped(
    kv_head_idx: int,
    kv_head_ratio: int,
    head_dim: int,
    rank: int,
    W_v: Tensor,
    W_o: Tensor,
    sqrt_C: Tensor,
    inv_sqrt_C: Tensor,
    new_heads_V: list[Tensor],
    new_heads_O: list[Tensor],
    slice_dims=True,
    arch="opt",
):
    # head_start_idx, head_end_idx
    kv_head_start, kv_head_end = kv_head_idx * head_dim, (kv_head_idx + 1) * head_dim

    V_head = W_v[kv_head_start:kv_head_end, :].to(dtype=dtype_p, device=d2)

    U, _S, V = torch.linalg.svd(sqrt_C @ V_head.T, full_matrices=False)

    S = torch.diag(_S)  # [head_dims, head_dims]

    compressed_v: Tensor = inv_sqrt_C @ U[:, :rank]
    if slice_dims:
        new_heads_V.append(compressed_v.T)
    else:
        V_new = torch.zeros_like(V_head).to(dtype=W_o.dtype, device=W_o.device)
        V_new[:rank, :] = compressed_v.T

        W_v[:, kv_head_start:kv_head_end].data.copy_(V_new)

    for j in range(kv_head_ratio):
        head_idx = (kv_head_idx * kv_head_ratio) + j
        head_s, head_e = head_idx * head_dim, (head_idx + 1) * head_dim

        O_head = W_o[:, head_s:head_e].to(dtype=dtype_p, device=d2)

        compressed_o = S[:rank, :rank] @ V[:rank, :] @ O_head.T

        if slice_dims:
            new_heads_O.append(compressed_o.T)
        else:
            O_new = torch.zeros_like(O_head).to(dtype=W_o.dtype, device=W_o.device)
            O_new[:, :rank] = compressed_o.T

            W_o[:, head_s:head_e].data.copy_(O_new)


@torch.no_grad()
def compress_head(
    head_idx: int,
    head_dim: int,
    rank_i: int,
    W_v: Tensor,
    W_o: Tensor,
    sqrt_C: Tensor,
    inv_sqrt_C: Tensor,
    new_heads_V: list[Tensor],
    new_heads_O: list[Tensor],
    slice_dims=True,
    arch="opt",
):
    # head_start_idx, head_end_idx
    head_s, head_e = head_idx * head_dim, (head_idx + 1) * head_dim
    # head_dims = head_dims, d_model = hidden dimensions
    # V_head [head_dims, d_model], O_head [d_model, head_dims]
    # sqrt_C [d_model, d_model]

    V_head = W_v[head_s:head_e, :].to(dtype=dtype_p, device=d2)
    O_head = W_o[:, head_s:head_e].to(dtype=dtype_p, device=d2)

    # V_head needs to be transposed because we expect a tensor of (in_features, out_features)
    # but torch stores weights as (out_features, in_features)
    U, _S, V = torch.linalg.svd(sqrt_C @ V_head.T, full_matrices=False)

    # sqrt_C @ V_head.T [d_model, head_dims]
    # U [d_model, head_dims], S [head_dims, head_dims], V [head_dims, head_dims]
    S = torch.diag(_S)  # [head_dims, head_dims]

    A = S @ V @ O_head.T  # once again transpose O_head
    U_p, _S_p, V_p = torch.linalg.svd(A, full_matrices=True)

    S_p = torch.diag(_S_p)

    # A [head_dims, d_model]
    # U_p [head_dims, head_dims], S_p [d_model, d_model], V_p [d_model, d_model]

    # [d_model, d_model] @ [d_model, head_dims] @ [head_dims, d_model] =
    # = [d_model, head_dims] @ [head_dims, head_dims] =
    # = [d_model, head_dims]
    compressed_v: Tensor = (inv_sqrt_C @ U @ U_p)[:, :rank_i]
    # [d_model, d_model] @ [d_model, d_model]
    compressed_o: Tensor = S_p[:rank_i, :rank_i] @ V_p[:rank_i, :]

    # compressed_v = compressed_v.T
    # compressed_o = compressed_o.T

    if slice_dims:
        new_heads_V.append(compressed_v.T)
        new_heads_O.append(compressed_o.T)
    else:
        V_new = torch.zeros_like(V_head).to(dtype=W_v.dtype, device=W_v.device)
        O_new = torch.zeros_like(O_head).to(dtype=W_o.dtype, device=W_o.device)

        # Retranspose the output because we transposed the input
        V_new[:rank_i, :] = compressed_v.T
        O_new[:, :rank_i] = compressed_o.T

        W_v[head_s:head_e, :].data.copy_(V_new)
        W_o[:, head_s:head_e].data.copy_(O_new)
