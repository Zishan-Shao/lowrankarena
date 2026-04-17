#coding:utf8
"""
Paper-derived v2-style heterogeneous reimplementation.

Implementation boundary:
- This file is a local reimplementation layered on top of `SVDLLM_v1_hetero`.
- It is not claimed to be a line-for-line equivalent of any upstream official
  public main-branch implementation.
- Relative to `SVDLLM_v1_hetero`, the intended v2 delta here is the profiling
  path: use raw X^T X plus a symmetric matrix square-root instead of the
  Cholesky-based v1 whitening path.
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

from SVDLLM_v1_hetero import (  # noqa: E402
    IMPLEMENTATION_BOUNDARY as V1_IMPLEMENTATION_BOUNDARY,
    IMPLEMENTATION_STATUS as V1_IMPLEMENTATION_STATUS,
    _get_layers,
    allocate_weight_type_keep_ratios,
    apply_module_keep_ratios,
)
from SVDLLM import (  # noqa: E402
    _maybe_release_guard,
    _maybe_reserve_guard,
    profle_svdllm_low_resource as _profile_v1_low_resource,
)
from utils.model_utils import find_layers  # noqa: E402


IMPLEMENTATION_STATUS = "local_paper_derived_v2_reimplementation"
BASELINE_IMPLEMENTATION_STATUS = V1_IMPLEMENTATION_STATUS
BASELINE_IMPLEMENTATION_BOUNDARY = V1_IMPLEMENTATION_BOUNDARY
IMPLEMENTATION_BOUNDARY = (
    "This module is a paper-derived local reimplementation built on top of "
    "SVDLLM_v1_hetero. It is not claimed to be a line-for-line equivalent of "
    "an upstream official public main-branch implementation."
)


def _sqrtm_svd_spd(mat: torch.Tensor, eps: float) -> torch.Tensor:
    if mat.dim() != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got {tuple(mat.shape)}")
    mat = (mat + mat.t()) * 0.5
    w, Q = torch.linalg.eigh(mat)
    w = torch.clamp(w, min=float(eps))
    return (Q * torch.sqrt(w)) @ Q.t()


def _svd_sqrtm_with_fallback(mat: torch.Tensor, dev) -> torch.Tensor:
    try:
        work = mat.to(dev, dtype=torch.float64)
        dmean = work.diag().abs().mean().item()
        eps = 1e-6 * (dmean if dmean > 0 else 1.0)
        return _sqrtm_svd_spd(work, eps=eps).cpu()
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
        work = mat.to("cpu", dtype=torch.float64)
        dmean = work.diag().abs().mean().item()
        eps = 1e-6 * (dmean if dmean > 0 else 1.0)
        return _sqrtm_svd_spd(work, eps=eps).cpu()


def _convert_cholesky_profile_to_v2_scaling(profiling_mat, dev):
    converted = {}
    print("Converting Cholesky whitening factors into symmetric sqrt(X^T X) factors...")
    for layer_idx, layer_profile in tqdm(profiling_mat.items()):
        converted_layer = {}
        for name, factor in layer_profile.items():
            factor = factor.detach().to(dtype=torch.float64)
            raw_second_moment = factor.matmul(factor.t())
            converted_layer[name] = _svd_sqrtm_with_fallback(raw_second_moment, dev=dev)
            del raw_second_moment
            if str(dev).startswith("cuda"):
                torch.cuda.empty_cache()
        converted[layer_idx] = converted_layer
    return converted


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
        "Start obtaining the whitening matrix (raw XTX, local paper-derived path)..."
        if raw_xtx
        else "Start obtaining the whitening matrix (centered covariance, local fallback)..."
    )
    print(msg)

    def hook(module, input, output):
        x = input[0].detach()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() != 2:
            x = x.view(-1, x.shape[-1])
        x = x.to(dtype=torch.float64, device=dev)
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
                module._acc = torch.zeros((in_f, in_f), dtype=torch.float64, device=dev)
            else:
                module._second = torch.zeros((in_f, in_f), dtype=torch.float64, device=dev)
                module._mean = torch.zeros((in_f,), dtype=torch.float64, device=dev)
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
    print("Start SVD sqrt factorization (no Cholesky)...")
    for i in tqdm(range(len(layers))):
        _maybe_release_guard(gpu_guard, f"v2 profiling factorization layer {i}")
        layer_profile = {}
        subset = find_layers(layers[i])
        for n in subset:
            if raw_xtx:
                if not hasattr(subset[n], "_acc"):
                    continue
                mat = subset[n]._acc.to(dev)
                dmean = mat.diag().abs().mean().item()
                eps = 1e-6 * (dmean if dmean > 0 else 1.0)
                scaling = _sqrtm_svd_spd(mat, eps=eps)
                subset[n]._acc = None
            else:
                if not hasattr(subset[n], "_second"):
                    continue
                second = subset[n]._second.to(dev)
                mean = subset[n]._mean.to(dev)
                count = max(int(subset[n]._count), 1)
                cov = second / count - torch.outer(mean / count, mean / count)
                cov = (cov + cov.t()) * 0.5
                dmean = cov.diag().abs().mean().item()
                eps = 1e-6 * (dmean if dmean > 0 else 1.0)
                scaling = _sqrtm_svd_spd(cov, eps=eps)
                subset[n]._second = None
                subset[n]._mean = None
                subset[n]._count = 0

            layer_profile[n] = scaling.cpu()
            del scaling
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
    previous_flag = os.environ.get("SVDLLM_COMPAT_WHITENING")
    try:
        os.environ["SVDLLM_COMPAT_WHITENING"] = "1" if raw_xtx else "0"
        cholesky_profile = _profile_v1_low_resource(
            name,
            model,
            calib_loader,
            dev,
            gpu_guard=gpu_guard,
            low_resource_factor_device=low_resource_factor_device,
            low_resource_activation_device=low_resource_activation_device,
        )
    finally:
        if previous_flag is None:
            os.environ.pop("SVDLLM_COMPAT_WHITENING", None)
        else:
            os.environ["SVDLLM_COMPAT_WHITENING"] = previous_flag

    return _convert_cholesky_profile_to_v2_scaling(cholesky_profile, dev=dev)


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
