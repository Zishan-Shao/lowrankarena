#coding:utf8
"""SVD-LLM v2 utilities + low-resource profiling + (optional) timing/GPU-memory instrumentation.

This file is a drop-in extension of your SVDLLM_v2 code:
- keeps the same algorithm
- adds a low-resource (layer-by-layer) profiler
- adds optional timing + GPU memory snapshots (no behavior change unless you pass timing lists)

The instrumentation style matches your DF-SVD instrumented script (CUDA sync + peak mem snapshots).
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.model_utils import find_layers
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTDecoderLayer

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)


# -------------------------
# Instrumentation helpers (copied style from df_svd_instrumented.py)
# -------------------------

def _cuda_sync(device: torch.device) -> None:
    """Synchronize CUDA to make wall-clock timing accurate."""
    if device.type == "cuda":
        try:
            torch.cuda.synchronize(device)
        except Exception:
            torch.cuda.synchronize()


def _reset_cuda_peak_stats(device: torch.device) -> None:
    """Reset CUDA peak memory stats for the current device."""
    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            torch.cuda.reset_peak_memory_stats()


def _cuda_mem_snapshot(device: torch.device) -> Dict[str, int]:
    """Return a snapshot of (current + peak) CUDA memory stats in bytes."""
    if device.type != "cuda":
        return {}
    try:
        free, total = torch.cuda.mem_get_info(device)
    except Exception:
        free, total = torch.cuda.mem_get_info()
    try:
        alloc = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_alloc = torch.cuda.max_memory_allocated(device)
        max_reserved = torch.cuda.max_memory_reserved(device)
    except Exception:
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_alloc = torch.cuda.max_memory_allocated()
        max_reserved = torch.cuda.max_memory_reserved()
    return {
        "alloc_bytes": int(alloc),
        "reserved_bytes": int(reserved),
        "max_alloc_bytes": int(max_alloc),
        "max_reserved_bytes": int(max_reserved),
        "free_bytes": int(free),
        "total_bytes": int(total),
    }


# -------------------------
# Core math helpers
# -------------------------

def _compat_enabled(key: str, default: bool = False) -> bool:
    """Check compatibility flags from environment.

    Supports SVDLLM_COMPAT_ALL=1 to enable everything, or per-feature toggles:
    SVDLLM_COMPAT_WHITENING, _RANKS, _ATTENTION.
    """
    if os.getenv("SVDLLM_COMPAT_ALL", "0") != "0":
        return True
    return os.getenv(f"SVDLLM_COMPAT_{key.upper()}", "1" if default else "0") != "0"


def _sqrtm_svd_spd(mat: torch.Tensor, eps: float) -> torch.Tensor:
    """Symmetric matrix square-root via eigendecomposition with eigenvalue clipping."""
    if mat.dim() != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got {tuple(mat.shape)}")
    mat = (mat + mat.t()) * 0.5
    w, Q = torch.linalg.eigh(mat)
    w = torch.clamp(w, min=float(eps))
    return (Q * torch.sqrt(w)) @ Q.t()


def truncated_svd(W: torch.Tensor, rank: int):
    """Compute full (economy) SVD, then truncate to the top-k singular values."""
    if rank <= 0:
        raise ValueError("rank must be positive.")
    U, S, VT = torch.linalg.svd(W, full_matrices=False)
    k = min(rank, S.shape[0])
    return U[:, :k], S[:k], VT[:k, :]


def randomized_svd(W: torch.Tensor, rank: int, niter: int = 2, oversample: int = 5):
    """Approximate top-k SVD via randomized low-rank factorization."""
    if rank <= 0:
        raise ValueError("rank must be positive.")
    q = min(rank + max(0, int(oversample)), min(W.shape))
    try:
        U, S, V = torch.svd_lowrank(W, q=q, niter=max(0, int(niter)))
    except Exception as e:
        raise RuntimeError(f"torch.svd_lowrank failed: {e}")
    k = min(rank, S.shape[0])
    return U[:, :k], S[:k], V[:, :k].T


# -------------------------
# Original (high-memory) profiler (kept as-is)
# -------------------------

@torch.no_grad()
def profle_svdllm(name, model, calib_loader, dev):
    """SVD-LLM v2 style profiling: compute whitening factor via symmetric sqrt (no Cholesky).

    NOTE: This version allocates an in_features x in_features accumulator for EVERY nn.Linear
    in the model at once. For LLaMA/Llama-2, that is likely to OOM.

    Prefer `profle_svdllm_low_resource_v2`.
    """
    name_l = str(name).lower()
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    is_opt = ("opt" in name_l) or (model_type == "opt")
    is_llama_like = any(tok in name_l for tok in ("llama", "mistral", "vicuna", "qwen")) or (
        model_type in ("llama", "mistral") or model_type.startswith("qwen")
    )
    if is_opt:
        layers = model.model.decoder.layers
    elif is_llama_like:
        layers = model.model.layers
    else:
        raise ValueError(f"Unsupported model family for profiling: name={name} model_type={model_type}")

    dev = torch.device(dev)
    model = model.to(dev)
    prev_cache = getattr(model.config, "use_cache", False)
    try:
        model.config.use_cache = False
    except Exception:
        pass

    compat_whitening = _compat_enabled("whitening", False)
    msg = (
        "Start obtaining the whitening matrix (raw XTX, official compat)..."
        if compat_whitening
        else "Start obtaining the whitening matrix (centered covariance)..."
    )
    print(msg)

    def hook(module, input, output):
        x = input[0].detach()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() == 2:
            pass
        else:
            x = x.view(-1, x.shape[-1])
        x = x.to(dtype=torch.float64, device=dev)
        if compat_whitening:
            module._acc += x.t().matmul(x)
        else:
            module._second += x.t().matmul(x)
            module._mean += x.sum(dim=0)
            module._count += x.shape[0]
        del x, output
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    handles = []
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_f = module.in_features
            if compat_whitening:
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
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    model = model.cpu()
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for n in subset:
            if compat_whitening and hasattr(subset[n], "_acc"):
                subset[n]._acc = subset[n]._acc.cpu()
            elif hasattr(subset[n], "_second"):
                subset[n]._second = subset[n]._second.cpu()
                subset[n]._mean = subset[n]._mean.cpu()

    profiling_mat = {}
    print("Start SVD sqrt factorization (no Cholesky)...")
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        subset = find_layers(layers[i])
        for n in subset:
            if compat_whitening:
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
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        profiling_mat[i] = layer_profile

    try:
        model.config.use_cache = prev_cache
    except Exception:
        pass
    return profiling_mat


# ---------------------------------------------------------------------------
# Low-resource (layer-by-layer) V2 profiling (INSTRUMENTED)
# ---------------------------------------------------------------------------

class _CatcherExit(Exception):
    """Internal control-flow exception to stop the forward pass early."""

    pass


def _tree_map(fn, x):
    """Apply fn to every tensor in a nested structure (tensor / list / tuple / dict)."""
    if torch.is_tensor(x):
        return fn(x)
    if isinstance(x, dict):
        return {k: _tree_map(fn, v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_tree_map(fn, v) for v in x)
    return x


@torch.no_grad()
def profle_svdllm_low_resource_v2(
    name,
    model,
    calib_loader,
    dev,
    *,
    max_batches: Optional[int] = None,
    stats_dtype: torch.dtype = torch.float32,
    store_act_dtype: torch.dtype = torch.float16,
    sqrt_dtype: torch.dtype = torch.float32,
    sqrt_on_gpu: Optional[bool] = None,
    # instrumentation
    tqdm_enabled: bool = True,
    timing_layers: Optional[List[Dict[str, Any]]] = None,
    timing_modules: Optional[List[Dict[str, Any]]] = None,
):
    """Memory-friendly SVD-LLM v2 profiling with optional timing/GPU-mem instrumentation.

    Returns:
      profiling_mat: Dict[int, Dict[str, torch.Tensor]] scaling matrices on CPU.

    If `timing_layers` / `timing_modules` are provided, this function appends records like:
      timing_layers += [{phase='profile', layer=i, forward_sec=..., sqrt_sec=..., total_sec=..., gpu_mem={...}}]
      timing_modules += [{phase='profile', layer=i, name=..., in_features=..., tokens=..., sqrt_sec=...}]
    """
    model.eval()

    dev = torch.device(dev)
    if sqrt_on_gpu is None:
        sqrt_on_gpu = dev.type == "cuda"

    name_l = str(name).lower()
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    is_opt = ("opt" in name_l) or (model_type == "opt")
    is_llama_like = any(tok in name_l for tok in ("llama", "mistral", "vicuna", "qwen")) or (
        model_type in ("llama", "mistral") or model_type.startswith("qwen")
    )
    if not (is_opt or is_llama_like):
        raise ValueError(f"Unsupported model family for low-resource profiling: name={name} model_type={model_type}")

    # Locate decoder layers and embedding stack.
    if is_opt:
        layers = model.model.decoder.layers
        embed_tokens = model.model.decoder.embed_tokens
        embed_positions = getattr(model.model.decoder, "embed_positions", None)
        project_in = getattr(model.model.decoder, "project_in", None)
        layernorm_embedding = getattr(model.model.decoder, "layernorm_embedding", None)
    else:
        layers = model.model.layers
        embed_tokens = model.model.embed_tokens
        embed_positions = None
        project_in = None
        layernorm_embedding = None

    # Disable KV cache during profiling.
    prev_cache = getattr(model.config, "use_cache", False)
    try:
        model.config.use_cache = False
    except Exception:
        pass

    compat_whitening = _compat_enabled("whitening", False)
    print(
        "Low-resource profiling (V2 sqrtm/eigh). "
        + ("Using raw XTX (compat)..." if compat_whitening else "Using centered covariance...")
    )

    # ---------------------------------------------------------------------
    # 1) Cache the inputs to the first decoder layer with a Catcher.
    # ---------------------------------------------------------------------
    model = model.cpu()

    # Move only the pre-layer0 embedding stack to GPU.
    embed_tokens = embed_tokens.to(dev)
    if embed_positions is not None:
        embed_positions = embed_positions.to(dev)
    if project_in is not None and isinstance(project_in, nn.Module):
        project_in = project_in.to(dev)
    if layernorm_embedding is not None and isinstance(layernorm_embedding, nn.Module):
        layernorm_embedding = layernorm_embedding.to(dev)

    cached_inps: List[torch.Tensor] = []
    cached_kwargs: List[Dict[str, Any]] = []

    def _cpu_store_tensor(t: torch.Tensor) -> torch.Tensor:
        # Keep ints/bools as-is; store floats in store_act_dtype to save RAM.
        if t.is_floating_point():
            return t.detach().to(dtype=store_act_dtype).cpu()
        return t.detach().cpu()

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, **kwargs):
            cached_inps.append(_cpu_store_tensor(hidden_states))
            kw: Dict[str, Any] = {}
            for k, v in kwargs.items():
                if torch.is_tensor(v):
                    kw[k] = _cpu_store_tensor(v)
                else:
                    kw[k] = _tree_map(lambda z: _cpu_store_tensor(z) if torch.is_tensor(z) else z, v)
            cached_kwargs.append(kw)
            raise _CatcherExit()

    # Wrap layer0
    first_layer = layers[0]
    layers[0] = Catcher(first_layer).to(dev)

    if tqdm_enabled:
        it_cache = tqdm(calib_loader, desc="Caching layer0 inputs")
    else:
        it_cache = calib_loader

    n = 0
    for batch in it_cache:
        if max_batches is not None and n >= int(max_batches):
            break
        n += 1
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        try:
            model(**batch)
        except _CatcherExit:
            pass

    # Restore layer0, free embedding GPU memory.
    layers[0] = first_layer.cpu()
    embed_tokens = embed_tokens.cpu()
    if embed_positions is not None:
        embed_positions = embed_positions.cpu()
    if project_in is not None and isinstance(project_in, nn.Module):
        project_in = project_in.cpu()
    if layernorm_embedding is not None and isinstance(layernorm_embedding, nn.Module):
        layernorm_embedding = layernorm_embedding.cpu()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    if len(cached_inps) == 0:
        raise RuntimeError("No calibration batches were cached. Check calib_loader and model forward.")

    # ---------------------------------------------------------------------
    # 2) Layer-by-layer stats: hook only that layer's Linear modules.
    # ---------------------------------------------------------------------
    profiling_mat: Dict[int, Dict[str, torch.Tensor]] = {}
    print(f"Cached {len(cached_inps)} calibration batches. Profiling {len(layers)} layers...")

    def make_hook():
        def hook(mod: nn.Module, inp: Tuple[torch.Tensor, ...], out: Any):
            x = inp[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            else:
                x = x.view(-1, x.shape[-1])
            x = x.to(dtype=stats_dtype)

            if compat_whitening:
                mod._acc.add_(x.t().matmul(x))
            else:
                mod._second.add_(x.t().matmul(x))
                mod._mean.add_(x.sum(dim=0))
                mod._count += x.shape[0]
            return None

        return hook

    def _move_kwargs_to_device(kw: Dict[str, Any], device):
        return _tree_map(lambda z: z.to(device) if torch.is_tensor(z) else z, kw)

    # Ping-pong buffers of layer inputs/outputs (lists of CPU tensors)
    inps_batches = cached_inps
    kwargs_batches = cached_kwargs

    if tqdm_enabled:
        it_layers = tqdm(range(len(layers)), desc="Profiling layers")
    else:
        it_layers = range(len(layers))

    for i in it_layers:
        if timing_layers is not None and dev.type == "cuda":
            _cuda_sync(dev)
            _reset_cuda_peak_stats(dev)

        layer_total_t0 = time.perf_counter()

        layer = layers[i].to(dev)
        subset = find_layers(layer)

        handles = []
        for ln, lin in subset.items():
            if not isinstance(lin, nn.Linear):
                continue
            in_f = lin.in_features
            if compat_whitening:
                lin._acc = torch.zeros((in_f, in_f), dtype=stats_dtype, device=dev)
            else:
                lin._second = torch.zeros((in_f, in_f), dtype=stats_dtype, device=dev)
                lin._mean = torch.zeros((in_f,), dtype=stats_dtype, device=dev)
                lin._count = 0
            handles.append(lin.register_forward_hook(make_hook()))

        # Forward cached batches through this single layer
        if dev.type == "cuda":
            _cuda_sync(dev)
        t_fwd0 = time.perf_counter()

        outs_batches: List[torch.Tensor] = []
        for inp_cpu, kw_cpu in zip(inps_batches, kwargs_batches):
            inp = inp_cpu.to(dev)
            kw = _move_kwargs_to_device(kw_cpu, dev)
            out = layer(inp, **kw)
            if isinstance(out, tuple):
                out = out[0]
            outs_batches.append(out.detach().to(dtype=store_act_dtype).cpu())

        if dev.type == "cuda":
            _cuda_sync(dev)
        forward_sec = time.perf_counter() - t_fwd0

        for h in handles:
            h.remove()

        # Build scaling matrices
        sqrt_total_sec = 0.0
        layer_profile: Dict[str, torch.Tensor] = {}
        for ln, lin in subset.items():
            if not isinstance(lin, nn.Linear):
                continue

            tokens = None
            if compat_whitening:
                mat = lin._acc.detach().to(dtype=sqrt_dtype)
                lin._acc = None
            else:
                second = lin._second.detach().to(dtype=sqrt_dtype)
                mean = lin._mean.detach().to(dtype=sqrt_dtype)
                cnt = max(int(lin._count), 1)
                tokens = int(cnt)
                cov = second / cnt - torch.outer(mean / cnt, mean / cnt)
                cov = (cov + cov.t()) * 0.5
                mat = cov
                lin._second = None
                lin._mean = None
                lin._count = 0

            # Compute sqrtm either on GPU or CPU
            if not sqrt_on_gpu:
                mat = mat.cpu()

            dmean = mat.diag().abs().mean().item()
            eps = 1e-6 * (dmean if dmean > 0 else 1.0)

            # accurate timing for GPU eigh
            if sqrt_on_gpu and dev.type == "cuda":
                _cuda_sync(dev)
            t_s0 = time.perf_counter()
            scaling = _sqrtm_svd_spd(mat, eps=eps)
            if sqrt_on_gpu and dev.type == "cuda":
                _cuda_sync(dev)
            sqrt_sec = time.perf_counter() - t_s0

            sqrt_total_sec += float(sqrt_sec)
            layer_profile[ln] = scaling.detach().cpu()

            if timing_modules is not None:
                timing_modules.append(
                    {
                        "phase": "profile",
                        "layer": int(i),
                        "name": str(ln),
                        "in_features": int(lin.in_features),
                        "tokens": None if tokens is None else int(tokens),
                        "sqrt_sec": float(sqrt_sec),
                        "sqrt_on_gpu": bool(sqrt_on_gpu),
                        "stats_dtype": str(stats_dtype).replace("torch.", ""),
                        "sqrt_dtype": str(sqrt_dtype).replace("torch.", ""),
                        "compat_whitening": bool(compat_whitening),
                    }
                )

            del mat, scaling
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        profiling_mat[i] = layer_profile

        # Move layer back to CPU and free GPU
        layers[i] = layer.cpu()
        del layer
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        # Next layer input = this layer output
        inps_batches = outs_batches

        total_sec = time.perf_counter() - layer_total_t0
        if timing_layers is not None:
            rec: Dict[str, Any] = {
                "phase": "profile",
                "layer": int(i),
                "forward_sec": float(forward_sec),
                "sqrt_sec": float(sqrt_total_sec),
                "total_sec": float(total_sec),
            }
            if dev.type == "cuda":
                rec["gpu_mem"] = _cuda_mem_snapshot(dev)
            timing_layers.append(rec)

    # Restore cache flag
    try:
        model.config.use_cache = prev_cache
    except Exception:
        pass

    return profiling_mat


# ---------------------------------------------------------------------------
# Whitening + SVD compression (INSTRUMENTED)
# ---------------------------------------------------------------------------

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
    # instrumentation
    tqdm_enabled: bool = True,
    timing_layers: Optional[List[Dict[str, Any]]] = None,
    timing_modules: Optional[List[Dict[str, Any]]] = None,
):
    """Whitening + SVD compression with heterogeneous ratios for attention vs MLP.

    If `timing_layers` / `timing_modules` are provided, appends records like:
      timing_layers += [{phase='compress', layer=i, total_sec=..., gpu_mem={...}}]
      timing_modules += [{phase='compress', layer=i, name=n, svd_sec=..., inv_sec=..., ...}]
    """
    model.eval()
    attn_ratio = float(ratio) if attn_ratio is None else float(attn_ratio)
    mlp_ratio = float(ratio) if mlp_ratio is None else float(mlp_ratio)
    svd_method = (svd_method or "full").lower()

    dev = torch.device(dev)

    model_name_l = str(model_name).lower()
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    is_opt = ("opt" in model_name_l) or (model_type == "opt")
    if is_opt:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    print(f"Start SVD decomposition after whitening (hetero): attn_ratio={attn_ratio}, mlp_ratio={mlp_ratio} ...")
    compat_ranks = _compat_enabled("ranks", False)
    compat_attn = _compat_enabled("attention", False)

    svd_time_total = 0.0

    if tqdm_enabled:
        it_layers = tqdm(range(len(layers)), desc="Compressing layers")
    else:
        it_layers = range(len(layers))

    for i in it_layers:
        if timing_layers is not None and dev.type == "cuda":
            _cuda_sync(dev)
            _reset_cuda_peak_stats(dev)

        layer_t0 = time.perf_counter()

        layer = layers[i]
        subset = find_layers(layer)

        # Replace Attn, MLP modules with different ratios
        is_qwen = ("qwen" in model_name_l) or model_type.startswith("qwen")
        if is_qwen:
            from component.svd_qwen import SVD_LlamaAttention as SVD_QwenAttention, SVD_LlamaMLP as SVD_QwenMLP

            svd_attn = SVD_QwenAttention(
                config=model.config,
                ratio=attn_ratio,
                compat_ranks=compat_ranks,
                compat_attention=compat_attn,
                base_attn=getattr(layer, "self_attn", None),
            )
            svd_attn.layer_idx = getattr(getattr(layer, "self_attn", None), "layer_idx", i)
            svd_mlp = SVD_QwenMLP(
                hidden_size=getattr(layer, "hidden_size", model.config.hidden_size),
                intermediate_size=model.config.intermediate_size,
                hidden_act=model.config.hidden_act,
                ratio=mlp_ratio,
                compat_ranks=compat_ranks,
            )
        elif ("llama" in model_name_l) or ("vicuna" in model_name_l) or (model_type == "llama"):
            svd_attn = SVD_LlamaAttention(
                config=model.config, ratio=attn_ratio, compat_ranks=compat_ranks, compat_attention=compat_attn
            )
            svd_attn.layer_idx = getattr(getattr(layer, "self_attn", None), "layer_idx", i)
            svd_mlp = SVD_LlamaMLP(
                hidden_size=getattr(layer, "hidden_size", model.config.hidden_size),
                intermediate_size=model.config.intermediate_size,
                hidden_act=model.config.hidden_act,
                ratio=mlp_ratio,
                compat_ranks=compat_ranks,
            )
        elif ("mistral" in model_name_l) or (model_type == "mistral"):
            svd_attn = SVD_MistralAttention(config=model.config, ratio=attn_ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=mlp_ratio)
        elif is_opt:
            # OPT has a single ratio in the decoder layer class; approximate with max of both.
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=max(attn_ratio, mlp_ratio))
        else:
            raise ValueError(f"Unsupported model name for whitening_hetero: {model_name}")

        for n in subset:
            mod_t0 = time.perf_counter()
            inv_sec = 0.0
            scale_sec = 0.0
            svd_sec = 0.0
            assemble_sec = 0.0
            inv_fallback = False

            orig_dtype = subset[n].weight.dtype
            W = subset[n].weight.data.to(dev, dtype=torch.float32)
            dtype = orig_dtype

            scaling_diag_matrix = profiling_mat[i][n].to(dev)

            # inv(scaling)
            if dev.type == "cuda":
                _cuda_sync(dev)
            t0 = time.perf_counter()
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception:
                inv_fallback = True
                scaling_diag_matrix = scaling_diag_matrix + 1e-6 * torch.eye(
                    scaling_diag_matrix.shape[0], device=dev, dtype=scaling_diag_matrix.dtype
                )
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            if dev.type == "cuda":
                _cuda_sync(dev)
            inv_sec = time.perf_counter() - t0

            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()

            # W_scale = W @ scaling
            if dev.type == "cuda":
                _cuda_sync(dev)
            t0 = time.perf_counter()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            if dev.type == "cuda":
                _cuda_sync(dev)
            scale_sec = time.perf_counter() - t0

            # Choose local ratio per module
            ln = str(n)
            if ("q_proj" in ln) or ("k_proj" in ln) or ("v_proj" in ln) or ("o_proj" in ln) or ("out_proj" in ln):
                local_ratio = attn_ratio
            elif ("gate_proj" in ln) or ("down_proj" in ln) or ("up_proj" in ln) or ("fc1" in ln) or ("fc2" in ln):
                local_ratio = mlp_ratio
            else:
                local_ratio = float(ratio)

            local_ratio = min(1.0, max(0.0, float(local_ratio)))
            max_rank = min(W.shape[0], W.shape[1])
            use_official_rank = compat_ranks or ("mistral" in model_name_l) or is_opt
            if use_official_rank:
                num_s_after_trunc = int(W.shape[0] * W.shape[1] * local_ratio / (W.shape[0] + W.shape[1]))
            else:
                num_s_after_trunc = int(max_rank * local_ratio)
            num_s_after_trunc = max(1, min(num_s_after_trunc, max_rank))

            # SVD
            if dev.type == "cuda":
                _cuda_sync(dev)
            t_svd_start = time.perf_counter()
            if svd_method == "randomized":
                U, S, VT = randomized_svd(W_scale, rank=num_s_after_trunc, niter=svd_niter, oversample=svd_oversample)
            elif svd_method == "truncated":
                U, S, VT = truncated_svd(W_scale, rank=num_s_after_trunc)
            else:
                U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            if dev.type == "cuda":
                _cuda_sync(dev)
            svd_sec = time.perf_counter() - t_svd_start
            svd_time_total += float(svd_sec)

            # assemble factors + assign
            t0 = time.perf_counter()
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
            truc_sigma = torch.diag(truc_s)
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)

            if is_opt:
                if "q_proj" in n:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, "self_attn", layer), "q_proj", None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, "self_attn", layer), "q_u_proj", None)
                    if prev_b is not None and getattr(prev_b, "bias", None) is not None:
                        svd_decoder.self_attn.q_u_proj.bias.data = prev_b.bias.data
                elif "k_proj" in n:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, "self_attn", layer), "k_proj", None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, "self_attn", layer), "k_u_proj", None)
                    if prev_b is not None and getattr(prev_b, "bias", None) is not None:
                        svd_decoder.self_attn.k_u_proj.bias.data = prev_b.bias.data
                elif "v_proj" in n:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, "self_attn", layer), "v_proj", None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, "self_attn", layer), "v_u_proj", None)
                    if prev_b is not None and getattr(prev_b, "bias", None) is not None:
                        svd_decoder.self_attn.v_u_proj.bias.data = prev_b.bias.data
                elif "out_proj" in n:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, "self_attn", layer), "out_proj", None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, "self_attn", layer), "out_u_proj", None)
                    if prev_b is not None and getattr(prev_b, "bias", None) is not None:
                        svd_decoder.self_attn.out_u_proj.bias.data = prev_b.bias.data
                elif "fc1" in n:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    prev_fc1 = getattr(layer, "fc1", None)
                    if prev_fc1 is None:
                        prev_fc1 = getattr(layer, "fc1_u_proj", None)
                    if prev_fc1 is not None and getattr(prev_fc1, "bias", None) is not None:
                        svd_decoder.fc1_u_proj.bias.data = prev_fc1.bias.data
                elif "fc2" in n:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    prev_fc2 = getattr(layer, "fc2", None)
                    if prev_fc2 is None:
                        prev_fc2 = getattr(layer, "fc2_u_proj", None)
                    if prev_fc2 is not None and getattr(prev_fc2, "bias", None) is not None:
                        svd_decoder.fc2_u_proj.bias.data = prev_fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in n:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in n:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in n:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in n:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn = svd_attn
                elif "gate_proj" in n:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in n:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in n:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp

            # Keep Linear metadata in sync with the new weight shapes
            def _sync_linear_meta(lin_mod: nn.Linear):
                if not isinstance(lin_mod, nn.Linear):
                    return
                lin_mod.in_features = lin_mod.weight.shape[1]
                lin_mod.out_features = lin_mod.weight.shape[0]
                if lin_mod.bias is not None and lin_mod.bias.numel() != lin_mod.out_features:
                    new_bias = lin_mod.bias.new_zeros(lin_mod.out_features)
                    sz = min(lin_mod.bias.numel(), lin_mod.out_features)
                    new_bias[:sz] = lin_mod.bias.data[:sz]
                    lin_mod.bias = nn.Parameter(new_bias, requires_grad=lin_mod.bias.requires_grad)

            _sync_linear_meta(subset[n])
            try:
                if not is_opt:
                    if "q_proj" in n:
                        _sync_linear_meta(svd_attn.q_u_proj)
                        _sync_linear_meta(svd_attn.q_v_proj)
                    elif "k_proj" in n:
                        _sync_linear_meta(svd_attn.k_u_proj)
                        _sync_linear_meta(svd_attn.k_v_proj)
                    elif "v_proj" in n:
                        _sync_linear_meta(svd_attn.v_u_proj)
                        _sync_linear_meta(svd_attn.v_v_proj)
                    elif "o_proj" in n:
                        _sync_linear_meta(svd_attn.o_u_proj)
                        _sync_linear_meta(svd_attn.o_v_proj)
                    elif "gate_proj" in n:
                        _sync_linear_meta(svd_mlp.gate_u_proj)
                        _sync_linear_meta(svd_mlp.gate_v_proj)
                    elif "down_proj" in n:
                        _sync_linear_meta(svd_mlp.down_u_proj)
                        _sync_linear_meta(svd_mlp.down_v_proj)
                    elif "up_proj" in n:
                        _sync_linear_meta(svd_mlp.up_u_proj)
                        _sync_linear_meta(svd_mlp.up_v_proj)
            except Exception:
                pass

            assemble_sec = time.perf_counter() - t0

            if timing_modules is not None:
                timing_modules.append(
                    {
                        "phase": "compress",
                        "layer": int(i),
                        "name": str(n),
                        "out_features": int(W.shape[0]),
                        "in_features": int(W.shape[1]),
                        "local_ratio": float(local_ratio),
                        "rank": int(num_s_after_trunc),
                        "use_official_rank": bool(use_official_rank),
                        "svd_method": str(svd_method),
                        "inv_fallback": bool(inv_fallback),
                        "inv_sec": float(inv_sec),
                        "scale_sec": float(scale_sec),
                        "svd_sec": float(svd_sec),
                        "assemble_sec": float(assemble_sec),
                        "total_sec": float(time.perf_counter() - mod_t0),
                    }
                )

            # free
            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT = truc_s = truc_u = truc_v = sqrtSigma = None
            del W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma

        if dev.type == "cuda":
            torch.cuda.empty_cache()

        layer_total_sec = time.perf_counter() - layer_t0
        if timing_layers is not None:
            rec: Dict[str, Any] = {"phase": "compress", "layer": int(i), "total_sec": float(layer_total_sec)}
            if dev.type == "cuda":
                rec["gpu_mem"] = _cuda_mem_snapshot(dev)
            timing_layers.append(rec)

    print(f"Total SVD time (all layers): {svd_time_total:.2f}s")
    return model


# =============================================================================
# Standalone STEP-1 CLI runner (embedded)
# =============================================================================

#!/usr/bin/env python3
# coding: utf-8
"""SVD-LLM v2 (sqrtm/eigh whitening) STEP-1 compression WITH timing + GPU memory stats.

This is the SVDLLM-v2 analog of your DF-SVD instrumented script:
- measures wall-clock time with CUDA synchronization
- captures per-layer peak GPU memory (reset per-layer)
- records per-module breakdown inside whitening+SVD
- writes a JSON timing file next to the saved checkpoint

Example (Llama-2-7B, paper-style parameter reduction=0.2 => keep=0.8):
  CUDA_VISIBLE_DEVICES=0 python -u svdllm_v2_step1_compress_instrumented.py \
    --model meta-llama/Llama-2-7b-hf \
    --ratio 0.2 --ratio_type reduction \
    --dataset wikitext2 --nsamples 256 --seq_len 2048 --batch_size 1 \
    --device cuda:0 --svd_method randomized \
    --save_path ./checkpoints/llama2_svdllmv2_r0.2.pt \
    --timing_file svdllmv2_timing.json

Notes:
- `--nsamples` counts *batches* (like your existing step1 script). Total sequences = nsamples * batch_size.
- Instrumentation is "no-op" for results; it only adds sync + JSON logging.
"""

import argparse
import json
import os
import random
import resource
import sys
import time
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    raise RuntimeError("This script requires transformers.") from e

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

# (functions are defined above in this one-file script)


# -------------------------
# Logging + instrumentation helpers (same style as df_svd_instrumented.py)
# -------------------------

def resolve_tqdm_enabled(mode: str) -> bool:
    mode = str(mode).lower().strip()
    if mode == "on":
        return True
    if mode == "off":
        return False
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


def _log(msg: str, *, tqdm_enabled: bool) -> None:
    if tqdm_enabled:
        try:
            from tqdm import tqdm as _tqdm

            _tqdm.write(msg)
            return
        except Exception:
            pass
    print(msg, flush=True)


def _cpu_maxrss_bytes() -> Optional[int]:
    try:
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(r)
        return int(r * 1024)
    except Exception:
        return None


def _model_size_bytes(model: torch.nn.Module) -> Dict[str, int]:
    param_bytes = 0
    param_count = 0
    for p in model.parameters():
        param_count += int(p.numel())
        param_bytes += int(p.numel()) * int(p.element_size())
    buffer_bytes = 0
    buffer_count = 0
    for b in model.buffers():
        buffer_count += int(b.numel())
        buffer_bytes += int(b.numel()) * int(b.element_size())
    return {
        "param_bytes": int(param_bytes),
        "buffer_bytes": int(buffer_bytes),
        "param_count": int(param_count),
        "buffer_count": int(buffer_count),
    }


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _timing_write(out_dir: str, filename: str, timing: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)
    return path


# -------------------------
# Args + dataset
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=str, required=True, help="HF model id or local path")
    p.add_argument("--dataset", type=str, default="wikitext2")
    p.add_argument("--nsamples", type=int, default=256, help="Number of calibration BATCHES")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=1)

    p.add_argument("--ratio", type=float, default=0.2)
    p.add_argument("--ratio_type", type=str, default="reduction", choices=["reduction", "keep"])

    p.add_argument("--attn_ratio", type=float, default=None)
    p.add_argument("--mlp_ratio", type=float, default=None)

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=3)

    p.add_argument("--load_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    p.add_argument("--stats_dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--sqrt_dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--store_act_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    p.add_argument("--sqrt_on_gpu", action="store_true")
    p.add_argument("--sqrt_on_cpu", action="store_true")

    p.add_argument("--svd_method", type=str, default="randomized", choices=["full", "randomized", "truncated"])
    p.add_argument("--svd_niter", type=int, default=2)
    p.add_argument("--svd_oversample", type=int, default=5)

    p.add_argument("--save_path", type=str, required=True)

    # instrumentation
    p.add_argument(
        "--timing_file",
        type=str,
        default="svdllmv2_timing.json",
        help="Write timing JSON to <dirname(save_path)>/<timing_file>",
    )
    p.add_argument("--tqdm", type=str, default="auto", choices=["auto", "on", "off"])

    return p.parse_args()


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    if s == "float64":
        return torch.float64
    raise ValueError(f"Unknown dtype string: {s}")


class _ChunkDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids = self.chunks[idx]
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def build_wikitext2_loader(tokenizer, nsamples_batches: int, seq_len: int, batch_size: int, seed: int) -> DataLoader:
    if load_dataset is None:
        raise RuntimeError("datasets is not available. Install datasets or use your repo's loader.")

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]

    total_sequences = nsamples_batches * batch_size
    max_start = int(input_ids.numel() - seq_len - 1)
    if max_start <= 0:
        raise RuntimeError(f"Dataset too small after tokenization for seq_len={seq_len}")

    rng = random.Random(seed)
    chunks = []
    for _ in range(total_sequences):
        start = rng.randint(0, max_start)
        chunk = input_ids[start : start + seq_len].clone()
        chunks.append(chunk)

    ds_chunks = _ChunkDataset(chunks)
    return DataLoader(ds_chunks, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)


def _phase_peak_from_layers(layers_list: list, phase: str) -> Tuple[int, int]:
    peak_alloc = 0
    peak_reserved = 0
    for rec in layers_list:
        if rec.get("phase") != phase:
            continue
        gm = rec.get("gpu_mem")
        if isinstance(gm, dict):
            peak_alloc = max(peak_alloc, int(gm.get("max_alloc_bytes", 0)))
            peak_reserved = max(peak_reserved, int(gm.get("max_reserved_bytes", 0)))
    return peak_alloc, peak_reserved


def main():
    args = parse_args()

    tqdm_enabled = resolve_tqdm_enabled(args.tqdm)
    if args.tqdm == "auto" and not tqdm_enabled:
        _log("[Info] tqdm disabled (stderr is not a TTY). Pass --tqdm on to force progress bars.", tqdm_enabled=False)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    # Ratios: paper uses "reduction". Our whitening_hetero expects "keep".
    def to_keep(r: Optional[float]) -> Optional[float]:
        if r is None:
            return None
        r = float(r)
        if args.ratio_type == "reduction":
            return 1.0 - r
        return r

    keep_ratio = float(to_keep(args.ratio))
    attn_keep = float(to_keep(args.attn_ratio)) if args.attn_ratio is not None else keep_ratio
    mlp_keep = float(to_keep(args.mlp_ratio)) if args.mlp_ratio is not None else keep_ratio

    out_dir = os.path.dirname(args.save_path) or "."

    run_start = time.perf_counter()
    timing: Dict[str, Any] = {
        "started_at": _now_iso(),
        "args": vars(args),
        "stages": [],
        "layers": [],
        "modules": [],
    }
    timing["args"]["tqdm_enabled"] = bool(tqdm_enabled)

    load_dtype = _dtype_from_str(args.load_dtype)

    # -------------------------
    # Load tokenizer
    # -------------------------
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    timing["stages"].append({"name": "load_tokenizer", "sec": time.perf_counter() - t0})

    # -------------------------
    # Load model
    # -------------------------
    t0 = time.perf_counter()
    _log(f"Loading base model: {args.model} (dtype={args.load_dtype}) ...", tqdm_enabled=tqdm_enabled)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=load_dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    _cuda_sync(dev)
    timing["stages"].append({"name": "load_model", "sec": time.perf_counter() - t0})

    # persistent footprint (pre)
    try:
        timing["model_size_before"] = _model_size_bytes(model)
    except Exception:
        pass

    # -------------------------
    # Calibration loader
    # -------------------------
    t0 = time.perf_counter()
    if args.dataset.lower() in ("wikitext2", "wikitext-2", "wikitext"):
        calib_loader = build_wikitext2_loader(
            tokenizer=tokenizer,
            nsamples_batches=args.nsamples,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    else:
        raise NotImplementedError(
            f"Dataset '{args.dataset}' not implemented in this standalone script. "
            f"Either add a loader or use your repo's existing data loader."
        )
    timing["stages"].append({"name": "build_calib_dataloader", "sec": time.perf_counter() - t0})

    stats_dtype = _dtype_from_str(args.stats_dtype)
    sqrt_dtype = _dtype_from_str(args.sqrt_dtype)
    store_act_dtype = _dtype_from_str(args.store_act_dtype)

    sqrt_on_gpu = True
    if args.sqrt_on_cpu:
        sqrt_on_gpu = False
    if args.sqrt_on_gpu:
        sqrt_on_gpu = True

    # -------------------------
    # Profiling
    # -------------------------
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    _cuda_sync(dev)
    t0 = time.perf_counter()
    profiling_mat = profle_svdllm_low_resource_v2(
        name=args.model,
        model=model,
        calib_loader=calib_loader,
        dev=dev,
        max_batches=args.nsamples,
        stats_dtype=stats_dtype,
        store_act_dtype=store_act_dtype,
        sqrt_dtype=sqrt_dtype,
        sqrt_on_gpu=sqrt_on_gpu,
        tqdm_enabled=tqdm_enabled,
        timing_layers=timing["layers"],
        timing_modules=timing["modules"],
    )
    _cuda_sync(dev)
    profile_sec = time.perf_counter() - t0
    timing["stages"].append({"name": "profile_whitening", "sec": float(profile_sec)})

    # -------------------------
    # Compression
    # -------------------------
    _log(
        f"Compressing with keep ratios: keep={keep_ratio:.4f}, attn_keep={attn_keep:.4f}, mlp_keep={mlp_keep:.4f} ...",
        tqdm_enabled=tqdm_enabled,
    )

    model = model.cpu()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    _cuda_sync(dev)
    t0 = time.perf_counter()
    model = whitening_hetero(
        model_name=args.model,
        model=model,
        profiling_mat=profiling_mat,
        ratio=keep_ratio,
        dev=dev,
        attn_ratio=attn_keep,
        mlp_ratio=mlp_keep,
        svd_method=args.svd_method,
        svd_niter=args.svd_niter,
        svd_oversample=args.svd_oversample,
        tqdm_enabled=tqdm_enabled,
        timing_layers=timing["layers"],
        timing_modules=timing["modules"],
    )
    _cuda_sync(dev)
    compress_sec = time.perf_counter() - t0
    timing["stages"].append({"name": "compress_whitening_svd", "sec": float(compress_sec)})

    # persistent footprint (post)
    try:
        timing["model_size_after"] = _model_size_bytes(model)
    except Exception:
        pass

    # -------------------------
    # Save
    # -------------------------
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    _log(f"Saving compressed model to: {args.save_path}", tqdm_enabled=tqdm_enabled)
    torch.save(model, args.save_path)
    timing["stages"].append({"name": "save_model", "sec": time.perf_counter() - t0})

    # -------------------------
    # Final timing summary
    # -------------------------
    _cuda_sync(dev)
    timing["ended_at"] = _now_iso()
    timing["total_sec"] = float(time.perf_counter() - run_start)

    # CPU RSS (best-effort)
    rss = _cpu_maxrss_bytes()
    if rss is not None:
        timing["cpu_maxrss_bytes"] = int(rss)

    # Phase-level GPU peaks from per-layer snapshots
    p_alloc, p_res = _phase_peak_from_layers(timing["layers"], "profile")
    c_alloc, c_res = _phase_peak_from_layers(timing["layers"], "compress")
    timing["gpu_peak_profile_alloc_bytes"] = int(p_alloc)
    timing["gpu_peak_profile_reserved_bytes"] = int(p_res)
    timing["gpu_peak_compress_alloc_bytes"] = int(c_alloc)
    timing["gpu_peak_compress_reserved_bytes"] = int(c_res)

    # Overall GPU peak
    overall_alloc = max(int(p_alloc), int(c_alloc))
    overall_res = max(int(p_res), int(c_res))
    timing["gpu_peak_alloc_bytes"] = int(overall_alloc)
    timing["gpu_peak_reserved_bytes"] = int(overall_res)

    if dev.type == "cuda":
        timing["gpu_mem_final"] = _cuda_mem_snapshot(dev)

    timing_path = _timing_write(out_dir, args.timing_file, timing)

    # user-friendly one-liner
    msg = (
        f"[Time] total={timing['total_sec']:.2f}s "
        f"profile={profile_sec:.2f}s compress={compress_sec:.2f}s "
        f"timing_json={timing_path}"
    )
    if dev.type == "cuda":
        ga = overall_alloc / (1024.0**3)
        gr = overall_res / (1024.0**3)
        msg += f" gpu_peak_alloc={ga:.2f}GB gpu_peak_reserved={gr:.2f}GB"
    _log(msg, tqdm_enabled=tqdm_enabled)


if __name__ == "__main__":
    main()
