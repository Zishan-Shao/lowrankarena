import numpy as np
import random
import logging

import torch
from torch.types import Tensor

from src.eval import load_calibration_texts
from src.model_utils import dtype_p, calib_device
from src.adapters.model_adapter import ModelAdapter

logger = logging.getLogger("MoDeGPT")

np.random.seed(1234)
random.seed(1234)


def load_calibs(
    adapter: ModelAdapter,
    n_samples: int,
    batch_size: int,
    dataset: str = "wikitext",
    load_calibs_from="",
    calibs_save_path="",
    target_layers: list[int] = [],  # optional param for compressing target layers
):

    cov_mlp, cov_q, cov_k, cov_x, bi_scores = __calibrate_model(
        adapter,
        n_samples=n_samples,
        batch_size=batch_size,
        dataset=dataset,
        target_layers=target_layers,
    )

    return cov_mlp, cov_q, cov_k, cov_x, bi_scores


@torch.no_grad()
def __calibrate_model(
    adapter: ModelAdapter,
    n_samples: int,
    batch_size: int,
    target_layers: list[int] = [],
    dataset="wikitext",
):

    model, tokenizer = adapter.model, adapter.tokenizer

    n_layers = adapter.n_layers
    n_experts = adapter.n_experts
    n_heads = adapter.n_heads
    head_dim = adapter.head_dim
    arch = adapter.arch
    transformer_blocks = adapter.get_transformer_blocks()

    if not target_layers:
        target_layers = [i for i in range(n_layers)]

    calc_bi_scores = True
    model.config.output_hidden_states = calc_bi_scores

    if adapter.calibs is None:
        adapter.calibs = load_calibration_texts(
            calib_size=n_samples,
            model=adapter.model,
            tokenizer=adapter.tokenizer,
            batch_size=batch_size,
            dataset=dataset,
        )

    logger.info(f"Detected architecture: {arch}")
    logger.info(f"target_layers = {target_layers}")
    logger.info("Calibrating model")

    bi_scores = [0.0 for _ in range(n_layers)]
    cov_mlp_list = [None for i in range(n_layers)]
    cov_k_list = [None for i in range(n_layers)]
    cov_q_list = [None for i in range(n_layers)]
    cov_x_list = [None for i in range(n_layers)]

    for i in target_layers:
        cov_mlp_list[i] = torch.zeros(
            adapter.get_n_inner(),
            adapter.get_n_inner(),
            dtype=dtype_p,
            device=calib_device,
        )

        cov_q_list[i] = torch.zeros(n_heads, head_dim, head_dim, dtype=dtype_p, device=calib_device)
        cov_k_list[i] = torch.zeros(
            adapter.n_kv_heads, head_dim, head_dim, dtype=dtype_p, device=calib_device
        )
        cov_x_list[i] = torch.zeros(
            adapter.d_model, adapter.d_model, dtype=dtype_p, device=calib_device
        )

    handles = []

    for layer_idx in target_layers:
        adapter.register_hooks(
            layer_idx,
            transformer_blocks[layer_idx],
            cov_mlp_list=cov_mlp_list,
            cov_q_list=cov_q_list,
            cov_k_list=cov_k_list,
            cov_x_list=cov_x_list,
            handles=handles,
            logger=logger,
        )

    model.eval()
    n_texts = 0
    for count, batch in enumerate(adapter.calibs):
        n_texts += len(batch)
        outputs = model(batch, output_hidden_states=calc_bi_scores)
        hidden_states = outputs.hidden_states
        for layer_idx in range(n_layers):
            x_in: Tensor = hidden_states[layer_idx].to(dtype_p)  # [B, T, D]
            x_out = hidden_states[layer_idx + 1].to(dtype_p)  # [B, T, D]

            bi_scores[layer_idx] += (
                torch.sum((1 - torch.cosine_similarity(x_in, x_out, dim=2)), dim=0).mean().item()
            )

            del x_in, x_out
        del hidden_states, outputs

        # logger.info(f"Completed {count + 1} of {len(texts)} batches")
    #####

    for h in handles:
        h.remove()

    for layer in range(n_layers):
        bi_scores[layer] /= n_texts

    if calc_bi_scores:
        adapter.bi_scores = bi_scores

    total_tokens = n_texts * 2048
    for layer_idx in target_layers:
        cov_mlp_list[layer_idx] /= total_tokens
        cov_x_list[layer_idx] /= total_tokens
        cov_k_list[layer_idx] /= total_tokens
        cov_q_list[layer_idx] /= total_tokens

    if logger:
        logger.info("Finished calibration and computed BI scores.")
    return cov_mlp_list, cov_q_list, cov_k_list, cov_x_list, bi_scores
