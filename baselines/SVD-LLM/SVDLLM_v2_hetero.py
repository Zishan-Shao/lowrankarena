#coding:utf8
"""
Paper-derived v2-style heterogeneous reimplementation.

Implementation boundary:
- This file is a local reimplementation of the v2 adaptive path.
- It is not claimed to be a line-for-line equivalent of any upstream official
  public main-branch implementation.
- Relative to the public uniform v1 path, the intended v2 delta here is:
  raw X^T X-style profiling plus adaptive/module-wise rank allocation.
"""

import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

from SVDLLM_adaptive_utils import (  # noqa: E402
    IMPLEMENTATION_BOUNDARY as ADAPTIVE_UTILS_IMPLEMENTATION_BOUNDARY,
    IMPLEMENTATION_STATUS as ADAPTIVE_UTILS_IMPLEMENTATION_STATUS,
    _get_layers,
    allocate_weight_type_keep_ratios,
    apply_module_keep_ratios,
)
from SVDLLM import _maybe_release_guard, _maybe_reserve_guard  # noqa: E402
from utils.model_utils import find_layers  # noqa: E402


IMPLEMENTATION_STATUS = "local_paper_derived_v2_reimplementation"
BASELINE_IMPLEMENTATION_STATUS = "public_v1_uniform_baseline"
BASELINE_IMPLEMENTATION_BOUNDARY = (
    "The baseline reference point for this module is the public uniform "
    "SVD-LLM v1 path rather than an adaptive v1 variant."
)
ADAPTIVE_UTILS_STATUS = ADAPTIVE_UTILS_IMPLEMENTATION_STATUS
ADAPTIVE_UTILS_BOUNDARY = ADAPTIVE_UTILS_IMPLEMENTATION_BOUNDARY
IMPLEMENTATION_BOUNDARY = (
    "This module is a paper-derived local reimplementation of the adaptive "
    "SVD-LLM v2 path. It is not claimed to be a line-for-line equivalent of "
    "an upstream official public main-branch implementation."
)


def _factorize_second_moment(mat: torch.Tensor, dev):
    if mat.dim() != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got {tuple(mat.shape)}")
    work = (mat + mat.t()) * 0.5
    try:
        work = work.to(dev, dtype=torch.float32)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
        work = work.to("cpu", dtype=torch.float32)

    eigvals, eigvecs = torch.linalg.eigh(work)
    eigvals = torch.clamp(eigvals, min=0.0)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals.index_select(0, order)
    eigvecs = eigvecs.index_select(1, order)

    root = torch.sqrt(eigvals)
    tol = max(1e-12, float(root.max().item()) * 1e-12 if root.numel() > 0 else 1e-12)
    inv_root = torch.where(root > tol, torch.reciprocal(root), torch.zeros_like(root))
    profile = {
        "u": eigvecs.cpu(),
        "root": root.cpu(),
        "inv_root": inv_root.cpu(),
    }
    del eigvals, eigvecs, root, inv_root, work
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache()
    return profile


@torch.no_grad()
def profle_svdllm(name, model, calib_loader, dev, *, raw_xtx: bool = True, gpu_guard=None):
    """Local v2-style profiling path.

    This function intentionally documents itself as a paper-derived local path,
    not as an official upstream equivalence claim.
    """
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers
    else:
        raise ValueError(f"Unsupported model name for profiling: {name}")

    model = model.to(dev)
    prev_cache = getattr(model.config, "use_cache", False)
    try:
        model.config.use_cache = False
    except Exception:
        pass

    msg = (
        "Start obtaining the v2 whitening stats (raw XTX)..."
        if raw_xtx
        else "Start obtaining the v2 whitening stats (centered covariance fallback)..."
    )
    print(msg)

    def hook(module, input, output):
        x = input[0].detach()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() != 2:
            x = x.view(-1, x.shape[-1])
        x = x.to(dtype=torch.float32, device=dev)
        if raw_xtx:
            module._acc += x.t().matmul(x)
        else:
            module._second += x.t().matmul(x)
            module._mean += x.sum(dim=0)
            module._count += x.shape[0]
        del x, output
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()

    handles = []
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_f = module.in_features
            if raw_xtx:
                module._acc = torch.zeros((in_f, in_f), dtype=torch.float32, device=dev)
            else:
                module._second = torch.zeros((in_f, in_f), dtype=torch.float32, device=dev)
                module._mean = torch.zeros((in_f,), dtype=torch.float32, device=dev)
                module._count = 0
            handles.append(module.register_forward_hook(hook))

    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model(**batch)

    for h in handles:
        h.remove()
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache()

    model = model.cpu()
    _maybe_reserve_guard(gpu_guard, "v2 whitening stats collected and model offloaded")
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for n in subset:
            if raw_xtx and hasattr(subset[n], "_acc"):
                subset[n]._acc = subset[n]._acc.cpu()
            elif hasattr(subset[n], "_second"):
                subset[n]._second = subset[n]._second.cpu()
                subset[n]._mean = subset[n]._mean.cpu()

    profiling_mat = {}
    print("Start factorizing second-moment matrices for v2 (no Cholesky)...")
    for i in tqdm(range(len(layers))):
        _maybe_release_guard(gpu_guard, f"v2 profiling factorization layer {i}")
        layer_profile = {}
        subset = find_layers(layers[i])
        for n in subset:
            if raw_xtx:
                if not hasattr(subset[n], "_acc"):
                    continue
                mat = subset[n]._acc.to(dev)
                subset[n]._acc = None
            else:
                if not hasattr(subset[n], "_second"):
                    continue
                second = subset[n]._second.to(dev)
                mean = subset[n]._mean.to(dev)
                count = max(int(subset[n]._count), 1)
                mat = second / count - torch.outer(mean / count, mean / count)
                subset[n]._second = None
                subset[n]._mean = None
                subset[n]._count = 0

            layer_profile[n] = _factorize_second_moment(mat, dev=dev)
            del mat
            if str(dev).startswith("cuda"):
                torch.cuda.empty_cache()

        profiling_mat[i] = layer_profile
        _maybe_reserve_guard(gpu_guard, f"v2 profiling factorization layer {i} finished")

    try:
        model.config.use_cache = prev_cache
    except Exception:
        pass
    return profiling_mat


profile_svdllm = profle_svdllm


@torch.no_grad()
def profle_svdllm_low_resource(
    name,
    model,
    calib_loader,
    dev,
    *,
    raw_xtx: bool = True,
    gpu_guard=None,
    low_resource_factor_device: str = "gpu",
    low_resource_activation_device: str = "auto",
):
    _maybe_release_guard(gpu_guard, "v2 low-resource profiling")
    if "opt" in name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            # Proxy attribute access to wrapped module so model.forward can
            # read layer-level attributes (e.g. Qwen3's attention_type).
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            idx = cache["i"]
            inps[idx] = inp
            cache["i"] += 1
            am = kwargs.get("attention_mask", None)
            pid = kwargs.get("position_ids", None)
            if am is not None:
                am_cpu = am.detach().cpu()
                cache["attention_mask"] = am_cpu if idx == 0 else torch.cat(
                    (cache["attention_mask"], am_cpu), dim=0
                ) if cache["attention_mask"] is not None else am_cpu
            if "opt" not in name and pid is not None:
                pid_cpu = pid.detach().cpu()
                cache["position_ids"] = pid_cpu if idx == 0 else torch.cat(
                    (cache["position_ids"], pid_cpu), dim=0
                ) if cache["position_ids"] is not None else pid_cpu
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        # rotary_emb is kept on dev — per-layer loop at line 324 also needs it
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_masks = cache["attention_mask"]
    position_ids = None if "opt" in name else cache["position_ids"]
    profiling_mat = {}
    print("Start low-resource factorization for v2 (no Cholesky)...")
    for i in tqdm(range(len(layers))):
        _maybe_release_guard(gpu_guard, f"v2 low-resource layer {i}")
        layer_profile = {}
        layer = layers[i].to(dev)
        subset = find_layers(layer)

        def hook(module, input, output):
            x = input[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            elif x.dim() != 2:
                x = x.view(-1, x.shape[-1])
            x = x.to(dtype=torch.float32)
            if raw_xtx:
                module._acc += x.t().matmul(x)
            else:
                module._second += x.t().matmul(x)
                module._mean += x.sum(dim=0)
                module._count += x.shape[0]
            del x, output
            if str(dev).startswith("cuda"):
                torch.cuda.empty_cache()

        handles = []
        for module_name, module in subset.items():
            in_f = module.in_features
            if raw_xtx:
                module._acc = torch.zeros((in_f, in_f), dtype=torch.float32, device=dev)
            else:
                module._second = torch.zeros((in_f, in_f), dtype=torch.float32, device=dev)
                module._mean = torch.zeros((in_f,), dtype=torch.float32, device=dev)
                module._count = 0
            handles.append(module.register_forward_hook(hook))

        for j in range(inps.shape[0]):
            if "opt" not in name:
                batch_inps = inps[j].unsqueeze(0)
                batch_position_ids = position_ids[j].unsqueeze(0).to(dev) if position_ids is not None else None
                am_j = attention_masks[j].unsqueeze(0).to(dev) if attention_masks is not None else None
                if hasattr(model.model, "rotary_emb") and batch_position_ids is not None:
                    position_embeddings = model.model.rotary_emb(batch_inps, batch_position_ids)
                else:
                    position_embeddings = None
                outs[j] = layer(
                    batch_inps,
                    attention_mask=am_j,
                    position_ids=batch_position_ids,
                    position_embeddings=position_embeddings,
                )[0]
            else:
                am_j = attention_masks[j].unsqueeze(0).to(dev) if attention_masks is not None else None
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=am_j,
                )[0]

        for h in handles:
            h.remove()

        for module_name, module in subset.items():
            if raw_xtx:
                mat = module._acc
                module._acc = None
            else:
                count = max(int(module._count), 1)
                mat = module._second / count - torch.outer(module._mean / count, module._mean / count)
                module._second = None
                module._mean = None
                module._count = 0
            layer_profile[module_name] = _factorize_second_moment(mat, dev=dev)
            del mat
            if str(dev).startswith("cuda"):
                torch.cuda.empty_cache()

        layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        inps = outs
        _maybe_reserve_guard(gpu_guard, f"v2 low-resource layer {i} finished")
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    return profiling_mat


profile_svdllm_low_resource = profle_svdllm_low_resource


@torch.no_grad()
def allocate_svdllm_v2_adaptive_keep_ratios(
    model_name,
    model,
    profiling_mat,
    target_reduction_ratio: float,
    dev,
    strict_paper_formula: bool = True,
):
    return allocate_weight_type_keep_ratios(
        model_name=model_name,
        model=model,
        profiling_mat=profiling_mat,
        target_reduction_ratio=target_reduction_ratio,
        dev=dev,
        strict_formula=strict_paper_formula,
        implementation_label="v2_hetero",
    )


@torch.no_grad()
def whitening_hetero(
    model_name,
    model,
    profiling_mat,
    ratio,
    dev,
    attn_ratio: float = None,
    mlp_ratio: float = None,
    svd_method: str = "full",
    svd_niter: int = 2,
    svd_oversample: int = 5,
    module_keep_ratios: Optional[Dict[Tuple[int, str], float]] = None,
    force_param_count_rank: bool = True,
    gpu_guard=None,
):
    return apply_module_keep_ratios(
        model_name=model_name,
        model=model,
        profiling_mat=profiling_mat,
        ratio=ratio,
        dev=dev,
        attn_ratio=attn_ratio,
        mlp_ratio=mlp_ratio,
        svd_method=svd_method,
        svd_niter=svd_niter,
        svd_oversample=svd_oversample,
        module_keep_ratios=module_keep_ratios,
        force_param_count_rank=force_param_count_rank,
        implementation_label="v2_hetero",
        gpu_guard=gpu_guard,
    )


@torch.no_grad()
def compress_model_adaptive(
    model_name,
    model,
    calib_loader,
    target_reduction_ratio: float,
    dev,
    strict_paper_formula: bool = True,
    raw_xtx: bool = True,
    svd_method: str = "full",
    svd_niter: int = 2,
    svd_oversample: int = 5,
    low_resource_profile: bool = False,
    low_resource_factor_device: str = "gpu",
    low_resource_activation_device: str = "auto",
    gpu_guard=None,
):
    """One-shot helper for the local v2-style heterogeneous path."""
    if low_resource_profile:
        profiling_mat = profle_svdllm_low_resource(
            model_name,
            model,
            calib_loader,
            dev,
            raw_xtx=raw_xtx,
            gpu_guard=gpu_guard,
            low_resource_factor_device=low_resource_factor_device,
            low_resource_activation_device=low_resource_activation_device,
        )
    else:
        profiling_mat = profle_svdllm(model_name, model, calib_loader, dev, raw_xtx=raw_xtx, gpu_guard=gpu_guard)
    module_keep_ratios, module_reduce_ratios, module_lmin = allocate_svdllm_v2_adaptive_keep_ratios(
        model_name=model_name,
        model=model,
        profiling_mat=profiling_mat,
        target_reduction_ratio=target_reduction_ratio,
        dev=dev,
        strict_paper_formula=strict_paper_formula,
    )
    whitening_hetero(
        model_name=model_name,
        model=model,
        profiling_mat=profiling_mat,
        ratio=1.0 - float(target_reduction_ratio),
        dev=dev,
        svd_method=svd_method,
        svd_niter=svd_niter,
        svd_oversample=svd_oversample,
        module_keep_ratios=module_keep_ratios,
        force_param_count_rank=True,
        gpu_guard=gpu_guard,
    )
    return module_keep_ratios, module_reduce_ratios, module_lmin, profiling_mat
