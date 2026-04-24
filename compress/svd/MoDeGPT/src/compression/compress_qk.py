import logging

import torch
from torch.types import Tensor

from src.compression_utils import sqrt_M
from src.model_utils import dtype_p, d1, d2

from src.adapters.model_adapter import ModelAdapter

logger = logging.getLogger("MoDeGPT")


#### NOTE: see OPTDecoderLayer.final_layer_norm
@torch.no_grad()
def compress_qk_svd(
    adapter: ModelAdapter,
    cov_x: list[Tensor],
    keep_ratios,
    rank=None,
    ridge_lambda=1,
    slice_dims=True,
):
    """
    QK compression using SVD instead of CR decomposition -- better performance for OPT models
    """
    model = adapter.model

    n_layers = adapter.n_layers
    n_heads = adapter.n_heads
    head_dim = adapter.head_dim

    for layer in range(n_layers):
        # try:
        keep_ratio = keep_ratios[layer]
        rank_i = int(head_dim * keep_ratio) if rank is None else rank
        rank_i = max(1, min(rank_i, head_dim))

        C = cov_x[layer].to(device="cuda", dtype=dtype_p)
        sqrt_C = sqrt_M(C, debug="QK covs:")
        inv_sqrt_C = torch.linalg.inv(sqrt_C)

        qk_weights = adapter.get_qk_components(layer_idx=layer)
        W_q: Tensor = qk_weights.query_proj.weight
        W_k: Tensor = qk_weights.key_proj.weight
        bias_q: Tensor = qk_weights.query_proj.bias
        bias_k: Tensor = qk_weights.key_proj.bias

        new_Q_heads = []
        new_K_heads = []
        bias_Q_heads = []
        bias_K_heads = []
        for h in range(n_heads):
            h_start = h * head_dim
            h_end = (h + 1) * head_dim

            Q_head: Tensor = W_q[h_start:h_end, :].to(C)
            K_head: Tensor = W_k[h_start:h_end, :].to(C)
            Q_head_bias: Tensor = bias_q[h_start:h_end].to(C)
            K_head_bias: Tensor = bias_k[h_start:h_end].to(C)

            u, s, v = torch.linalg.svd(sqrt_C @ Q_head.T, full_matrices=False)
            s = torch.diag(s)

            # print(f"u.shape = {u.shape}")
            # print(f"s.shape = {s.shape}")
            # print(f"v.shape = {v.shape}")

            u_p, s_p, v_p = torch.linalg.svd(s @ v @ K_head)

            Q = (inv_sqrt_C @ u @ u_p)[:, :rank_i]
            K = torch.diag(s_p[:rank_i]) @ v_p[:rank_i]

            # print(f"Q_head.shape = {Q_head.shape}")  # [d_model, d_model]
            # print(f"u.shape = {u.shape}")  # [d_model, d_model]
            # print(f"s.shape = {s.shape}")  # [d_model]
            # print(f"v.shape = {v.shape}")  # [d_model, d_model]
            # print(f"inv_sqrt_C.shape = {inv_sqrt_C.shape}")  # [d_model, d_model]

            # Q = inv_sqrt_C @ u[:, :rank_i]  # [d_model, rank_i]
            # K = v[:, :rank_i] @ torch.diag_embed(s[:rank_i])  # [d_model, rank_i]
            alpha = torch.sqrt(torch.abs(K).max() / torch.abs(Q).max())  # make Q K of similar scale
            # print(f"Q.shape = {Q.shape}")  # [d_model, d_model]
            Q = Q * alpha
            K /= alpha

            # print(f"Q.shape = {Q.shape}")
            # print(f"K.shape = {K.shape}")

            Q = Q.T  # [rank_i, d_model]
            K = K  # [rank_i, d_model]

            new_Q_heads.append(Q)
            new_K_heads.append(K)

            # Q.T [d_model, rank_i], Q_head.T [head_dim, ]
            # bias_proj_Q = Q_head.T @ Q_head_bias.view(-1, 1)
            # bias_proj_K = K_head.T @ K_head_bias.view(-1, 1)
            # print(f"bias_proj_Q.shape = {bias_proj_Q.shape}")
            # print(f"bias_proj_K.shape = {bias_proj_K.shape}")
            # new_bias_Q = torch.linalg.solve(Q, bias_proj_Q).view(-1)
            # new_bias_K = torch.linalg.solve(K, bias_proj_K).view(-1)

            scale = alpha * torch.diag_embed(torch.reciprocal(s_p[:rank_i]))  #
            # scale = 1 / alpha * torch.diag_embed(torch.reciprocal(s_p[:rank_i])) # makes scale too small, bias goes to 0
            # print(f"alpha = {alpha}")
            # print(f"scale.mean() = {scale.mean()}")
            # print(f"scale.shape = {scale.shape}")
            # print(f"u_p.shape = {u_p.shape}")
            # print(f"v_p.shape = {v_p.shape}")
            # [rank_i, rank_i] @ [rank_i, d_model] @ [d_model, rank_i]
            new_bias_Q: Tensor = scale @ v_p[:, :rank_i].T @ K_head.T @ Q_head_bias
            # print(f"old_bias_Q.mean = {Q_head_bias.mean(dim=0)}")
            # print(f"new_bias_Q.mean = {new_bias_Q.mean(dim=0)}")

            # new_bias_Q = (torch.pinverse(Q.T) @ Q_head.T @ Q_head_bias.view(-1, 1)).view(-1) # also makes bias go to zero
            new_bias_K = (torch.pinverse(K.T) @ K_head.T @ K_head_bias.view(-1, 1)).view(
                -1
            )  # also makes bias go to zero

            bias_Q_heads.append(new_bias_Q)
            bias_K_heads.append(new_bias_K)

            # Q_new = Q_head.T @ Sk  # dont trust this, gotta rethink about that
            # K_new = K_head.T @ Sk
            # bias_q_new = bias_q[h_start:h_end][topk_selector]
            # bias_k_new = bias_k[h_start:h_end][topk_selector]
            # new_Q_heads.append(Q_new.T)
            # new_K_heads.append(K_new.T)
            # bias_Q_heads.append(bias_q_new)
            # bias_K_heads.append(bias_k_new)

        ####

        slice_QK_dims(
            model=model,
            layer_idx=layer,
            new_heads_Q=new_Q_heads,
            new_heads_K=new_K_heads,
            new_bias_Q=bias_Q_heads,
            new_bias_K=bias_K_heads,
            bias=False,
        )

        if logger:
            logger.info(
                f"[QK] ✅ Layer {layer}: compressed to rank {rank_i} per head (CR-score + interpolation)"
            )


#### NOTE: see OPTDecoderLayer.final_layer_norm
@torch.no_grad()
def compress_qk(
    adapter: ModelAdapter,
    cov,
    keep_ratios,
    rank=None,
    slice_dims=True,
    target_layers: list[int] | None = None,
):

    if target_layers is None:
        target_layers = [i for i in range(adapter.n_layers)]

    cov_q_list, cov_k_list = cov  # List[List[Tensor]] for Q and K respectively

    model = adapter.model

    n_layers = adapter.n_layers
    head_dim = adapter.head_dim
    arch = adapter.arch

    rotary_masks = []
    for i in target_layers:
        # try:
        keep_ratio = keep_ratios[i]
        rank_i = int(head_dim * keep_ratio) if rank is None else rank
        rank_i = max(1, min(rank_i, head_dim))

        if arch == "llama" or "qwen" in arch:
            rank_i = rank_i - (rank_i % 2)
            rank_i = max(2, min(rank_i, head_dim))

        compress_layer(
            adapter,
            i,
            rank_i,
            cov_q_list=cov_q_list[i],
            cov_k_list=cov_k_list[i],
            rotary_masks=rotary_masks,
            slice_dims=slice_dims,
        )

        if logger:
            logger.info(
                f"[QK] ✅ Layer {i}: compressed to rank {rank_i} per head (CR-score + interpolation)"
            )

    if not slice_dims:
        rotary_masks = None
    return rotary_masks
    # except Exception as e:
    #     if logger:
    #         logger.error("Error: %s", e, exc_info=True)
    #         logger.warning(f"[QK] Compression failed at layer {i}: {e}")


@torch.no_grad()
def compress_layer(
    adapter: ModelAdapter,
    layer_idx: int,
    rank: int,
    cov_q_list: list[Tensor],
    cov_k_list: list[Tensor],
    rotary_masks: list[Tensor],
    slice_dims=True,
    bias=True,
):

    n_heads, head_dim, arch = adapter.n_heads, adapter.head_dim, adapter.arch
    n_kv_heads = adapter.n_kv_heads

    grouped = n_kv_heads != n_heads

    qk_weights = adapter.get_qk_components(layer_idx=layer_idx)
    W_q = qk_weights.query_proj.weight
    W_k = qk_weights.key_proj.weight
    bias_q = qk_weights.query_proj.bias
    bias_k = qk_weights.key_proj.bias

    bias = bias_q is not None

    new_Q_heads = []
    new_K_heads = []
    bias_Q_heads = []
    bias_K_heads = []
    layer_rotary_mask = []
    Wq_heads = W_q.view(n_heads, head_dim, -1)
    Wk_heads = W_k.view(n_kv_heads, head_dim, -1)
    for h in range(n_kv_heads):
        h_start = h * head_dim
        h_end = (h + 1) * head_dim

        if (arch == "llama" or "qwen" in arch) and grouped:
            compress_head_llama_grouped(
                kv_head_idx=h,
                kv_head_ratio=n_heads // n_kv_heads,
                cov_q_layer=cov_q_list,
                cov_k_layer=cov_k_list,
                Wq_heads=Wq_heads,
                Wk_heads=Wk_heads,
                Q_heads_out=new_Q_heads,
                K_heads_out=new_K_heads,
                layer_rotary_mask=layer_rotary_mask,
                rank=rank,
                ridge_lambda=adapter.config.ridge_qk,
            )
        elif arch == "llama":
            compress_head_llama(
                cov_q_list[h],
                cov_k_list[h],
                Wq_heads[h],
                Wk_heads[h],
                Q_heads_out=new_Q_heads,
                K_heads_out=new_K_heads,
                layer_rotary_mask=layer_rotary_mask,
                slice_dims=slice_dims,
                rank=rank,
            )
        elif arch == "opt":
            Q_head_bias: Tensor = bias_q[h_start:h_end]
            K_head_bias: Tensor = bias_k[h_start:h_end]
            compress_head_opt(
                C_q=cov_q_list[h],
                C_k=cov_k_list[h],
                Q_head=Wq_heads[h],
                K_head=Wk_heads[h],
                bias_Q_head=Q_head_bias,
                bias_K_head=K_head_bias,
                out_Q_heads=new_Q_heads,
                out_K_heads=new_K_heads,
                out_Q_bias=bias_Q_heads,
                out_K_bias=bias_K_heads,
                rank=rank,
            )
        else:
            raise NotImplementedError(
                "Most likely have to implement it compression for this model."
            )
    if (arch == "llama" or "qwen" in arch) and slice_dims:
        final = torch.tensor([], dtype=torch.int64, device="cuda")
        for mask_head in layer_rotary_mask:
            final = torch.cat((final, mask_head.to(device="cuda", dtype=torch.int64)), dim=0)

        final = final.reshape(n_kv_heads, -1)
        rotary_masks.append(final)

    Q_heads = torch.cat(new_Q_heads, dim=0).to(device="cuda", dtype=torch.bfloat16)
    K_heads = torch.cat(new_K_heads, dim=0).to(device="cuda", dtype=torch.bfloat16)

    weight_cache = {"q_proj": Q_heads, "k_proj": K_heads}

    adapter.save_layer(
        output_dir=adapter.config.temp_storage_dir,
        suffix="qk",
        weights=weight_cache,
        layer_idx=layer_idx,
    )

    # adapter.slice_qk_dims(
    #     layer_idx=layer_idx,
    #     new_heads_Q=new_Q_heads,
    #     new_heads_K=new_K_heads,
    #     new_bias_Q=bias_Q_heads,
    #     new_bias_K=bias_K_heads,
    #     bias=bias,
    # )


@torch.no_grad()
def compress_head_llama_grouped(
    kv_head_idx: int,
    kv_head_ratio: int,
    cov_q_layer: list[Tensor],
    cov_k_layer: list[Tensor],
    Wq_heads: Tensor,
    Wk_heads: Tensor,
    Q_heads_out: list[Tensor],
    K_heads_out: list[Tensor],
    layer_rotary_mask: list[Tensor],
    rank: int,
    slice_dims=True,
    ridge_lambda=1e-4,
):
    """
    llama uses RoPE so its a little different than for OPT.
    """

    q_head_idx = kv_head_idx * kv_head_ratio

    K_head = Wk_heads[kv_head_idx]
    Q_heads = Wq_heads[q_head_idx : q_head_idx + kv_head_ratio]

    head_dims = K_head.shape[0]

    group_score = torch.zeros(head_dims // 2).to(device=d2, dtype=dtype_p)

    sqrt_C_k = sqrt_M(
        cov_k_layer[kv_head_idx].to(device=d2, dtype=dtype_p), ridge_lambda=ridge_lambda
    )
    for cov_q_h in cov_q_layer[q_head_idx : q_head_idx + kv_head_ratio]:
        cov_q_h = cov_q_h.to(device=d2, dtype=dtype_p)
        sqrt_C_q = sqrt_M(cov_q_h)

        normed_q_r1 = torch.norm(sqrt_C_q[..., : head_dims // 2], dim=0)
        normed_q_r2 = torch.norm(sqrt_C_q[..., head_dims // 2 :], dim=0)
        normed_k_r1 = torch.norm(sqrt_C_k[..., : head_dims // 2], dim=0)
        normed_k_r2 = torch.norm(sqrt_C_k[..., head_dims // 2 :], dim=0)

        final_norm = normed_q_r1**2 * normed_k_r1**2 + normed_q_r2**2 * normed_k_r2**2

        group_score += final_norm

    group_score = torch.sqrt(group_score)

    topk = torch.topk(group_score, k=rank // 2).indices
    Sk_mask = torch.cat((topk, topk + (head_dims // 2)))

    if not slice_dims:
        new_Q_heads = torch.zeros_like(Q_heads)
        new_K_head = torch.zeros_like(K_head)
        new_Q_heads[:, Sk_mask, :] = Q_heads[:, Sk_mask, :]
        new_K_head[Sk_mask, :] = K_head[Sk_mask, :]
    else:
        new_Q_heads = Q_heads[:, Sk_mask, :]
        new_K_head = K_head[Sk_mask, :]

    K_heads_out.append(new_K_head)
    for new_Q_head in new_Q_heads:
        Q_heads_out.append(new_Q_head)

    layer_rotary_mask.append(Sk_mask)

    # return new_Q_head, new_K_head, Sk_mask


@torch.no_grad()
def compress_head_llama(
    C_q: Tensor,
    C_k: Tensor,
    Q_head: Tensor,
    K_head: Tensor,
    Q_heads_out: list[Tensor],
    K_heads_out: list[Tensor],
    layer_rotary_mask: list[Tensor],
    rank: int,
    slice_dims=True,
):
    """
    llama uses RoPE so its a little different than for OPT.
    """

    C_q = C_q.to(dtype=dtype_p, device=d2)
    C_k = C_k.to(dtype=dtype_p, device=d2)

    sqrt_C_q = sqrt_M(C_q)
    sqrt_C_k = sqrt_M(C_k)

    head_dims = Q_head.shape[0]

    normed_q_r1 = torch.norm(sqrt_C_q[..., : head_dims // 2], dim=0)  # norm for query rotary half 1
    normed_q_r2 = torch.norm(sqrt_C_q[..., head_dims // 2 :], dim=0)
    normed_k_r1 = torch.norm(sqrt_C_k[..., : head_dims // 2], dim=0)
    normed_k_r2 = torch.norm(sqrt_C_k[..., head_dims // 2 :], dim=0)  # norm for key rotary half 2

    final_norm = normed_q_r1**2 * normed_k_r1**2 + normed_q_r2**2 * normed_k_r2**2

    topk = torch.topk(final_norm, k=rank // 2).indices
    Sk_mask = torch.cat((topk, topk + (head_dims // 2)))

    if not slice_dims:
        new_Q_head = torch.zeros_like(Q_head).to(Q_head)
        new_K_head = torch.zeros_like(K_head).to(K_head)
        new_Q_head[Sk_mask, :] = Q_head[Sk_mask, :]
        new_K_head[Sk_mask, :] = K_head[Sk_mask, :]
    else:
        new_Q_head = Q_head[Sk_mask, :]
        new_K_head = K_head[Sk_mask, :]

    new_Q_head = new_Q_head.to(device="cpu", dtype=torch.bfloat16)
    new_K_head = new_K_head.to(device="cpu", dtype=torch.bfloat16)

    Q_heads_out.append(new_Q_head)
    K_heads_out.append(new_K_head)

    layer_rotary_mask.append(Sk_mask)


def compress_head_opt(
    C_q: Tensor,
    C_k: Tensor,
    Q_head: Tensor,
    K_head: Tensor,
    bias_Q_head: Tensor,
    bias_K_head: Tensor,
    out_Q_heads: list[Tensor],
    out_K_heads: list[Tensor],
    out_Q_bias: list[Tensor],
    out_K_bias: list[Tensor],
    rank: int,
):
    C_q = C_q.to(dtype=dtype_p, device=d2)
    C_k = C_k.to(dtype=dtype_p, device=d2)

    sqrt_C_q = sqrt_M(C_q)
    sqrt_C_k = sqrt_M(C_k)

    # symmetric matrix, dim doesnt matter
    norms_q = torch.linalg.vector_norm(sqrt_C_q, dim=0)
    norms_k = torch.linalg.vector_norm(sqrt_C_k, dim=0)
    scores = norms_q * norms_k

    # NOTE TO ME LATER::: Consider dims input into topk (does it work with vector norm above)
    topk = torch.topk(scores, k=rank).indices

    Q_new = Q_head[topk].to(device="cpu")
    K_new = K_head[topk].to(device="cpu")

    bias_q_new = bias_Q_head[topk]
    bias_k_new = bias_K_head[topk]
    out_Q_heads.append(Q_new)
    out_K_heads.append(K_new)
    out_Q_bias.append(bias_q_new)
    out_K_bias.append(bias_k_new)

    return Q_new, K_new, bias_q_new, bias_k_new
