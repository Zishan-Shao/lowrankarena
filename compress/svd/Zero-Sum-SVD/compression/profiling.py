"""
Profiling and whitening matrix computation for SVD-LLM compression.

Extracted from SVDLLM_sum0strategy_alllowrank_fullprune_eachstage_onescalaer_correction.py
"""

import torch
import torch.nn as nn
from torch.linalg import LinAlgError
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm

from utils.model_utils import LowRankLinear, find_target_modules_in_layer, get_layers
from utils.data_utils import get_calib_train_data
from compression.batch_utils import _model_forward


def svd_whitened_sorted(W: torch.Tensor, L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SVD of A = W @ L with descending singular values.
    Returns U, S, Vh on the same device as W.
    """
    A = W @ L
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    idx = torch.argsort(S, descending=True)
    return U[:, idx], S[idx], Vh[idx, :]


def prepare_calib_loader(args, tokenizer) -> List[Dict[str, torch.Tensor]]:
    """
    Return the base calibration data.

    If args.base_calib_data exists, reuse it.
    Otherwise, build it once from get_calib_train_data.
    """
    base = getattr(args, "base_calib_data", None)
    if base is not None:
        return base

    cali_data = get_calib_train_data(
        args.dataset,
        tokenizer,
        args.nsamples,
        seqlen=args.model_seq_len,
        seed=args.seed,
    )
    base_list = list(cali_data)
    args.base_calib_data = base_list
    return base_list


def compute_profiling_matrices(args, model, tokenizer, dev) -> Dict:
    """Compute profiling matrices via SVDLLM flow."""
    print("\n🔁 Profiling matrices not provided; computing via SVDLLM flow...")
    calib_loader = prepare_calib_loader(args, tokenizer)

    # Parse efficient_accumulate settings
    eff = getattr(args, "efficient_accumulate", False)
    chunk_tokens = int(getattr(args, "efficient_accumulate_chunk_tokens", 256))
    dtype_str = getattr(args, "efficient_accumulate_dtype", "bf16")
    if dtype_str == "bf16":
        eff_dtype = torch.bfloat16
    elif dtype_str == "fp16":
        eff_dtype = torch.float16
    else:
        eff_dtype = torch.float32

    if getattr(args, "profile_independently", False):
        profiling_mat = profle_svdllm_independent(
            args.model, model, calib_loader, dev,
            efficient_accumulate=eff,
            chunk_tokens=chunk_tokens,
            chunk_dtype=eff_dtype,
        )
        print("   ✓ Independent profiling matrices computed\n")
        return profiling_mat

    profiling_mat = profle_svdllm_shared(
        args.model, model, calib_loader, dev,
        efficient_accumulate=eff,
        chunk_tokens=chunk_tokens,
        chunk_dtype=eff_dtype,
    )
    print("   ✓ Shared profiling matrices computed\n")
    return profiling_mat


@torch.no_grad()
def profle_svdllm_shared(
    model_name: str,
    model,
    calib_loader,
    dev,
    eps: float = 1e-4,
    efficient_accumulate: bool = False,
    chunk_tokens: int = 256,
    chunk_dtype: torch.dtype = torch.bfloat16,
) -> Dict:
    """
    Compute shared input whitening matrices for all target modules.

    Uses forward hooks to accumulate covariance matrices during calibration,
    then computes Cholesky factors for whitening.

    Args:
        efficient_accumulate: If True, use chunked gram accumulation with CPU offload.
        chunk_tokens: Token-chunk size for efficient accumulation.
        chunk_dtype: Dtype used on GPU for chunked gram matmul.
    """
    layers = get_layers(model_name, model)

    model = model.to(dev).eval()
    print("Start obtaining shared input whitening matrices...")
    if efficient_accumulate:
        print(f"   Using efficient accumulation (chunk_tokens={chunk_tokens}, dtype={chunk_dtype})")

    C_in, N_in = {}, {}
    C_attn, N_attn = {}, {}
    C_mlp_in, N_mlp_in = {}, {}
    C_h, N_h = {}, {}
    handles = []

    def _accumulate(Cdict, Ndict, layer_idx, tensor):
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        B, T, D = tensor.shape
        flat = tensor.reshape(B * T, D).to(dev, dtype=torch.float32)
        gram = flat.transpose(0, 1) @ flat  # stays float32
        if layer_idx in Cdict:
            Cdict[layer_idx].add_(gram)
            Ndict[layer_idx] += B * T
        else:
            Cdict[layer_idx] = gram
            Ndict[layer_idx] = B * T

    def _accumulate_efficient(Cdict, Ndict, layer_idx, tensor):
        """Chunked gram accumulation with CPU offload to reduce GPU memory."""
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        B, T, D = tensor.shape

        # Allocate CPU gram once per layer
        if layer_idx not in Cdict:
            Cdict[layer_idx] = torch.zeros((D, D), device="cpu", dtype=torch.float32)
            Ndict[layer_idx] = 0

        # Flatten on GPU in smaller dtype
        flat = tensor.reshape(B * T, D).to(dev, dtype=chunk_dtype)

        # Chunk along token dimension to limit matmul peak
        for chunk in flat.split(chunk_tokens, dim=0):
            # Gram on GPU, then immediately offload
            gram_chunk = (chunk.transpose(0, 1) @ chunk).float().cpu()
            Cdict[layer_idx].add_(gram_chunk)

        Ndict[layer_idx] += B * T

    # Select accumulator function
    acc_fn = _accumulate_efficient if efficient_accumulate else _accumulate

    for layer_idx, layer in enumerate(layers):
        if hasattr(layer, "input_layernorm"):
            def tap_in(_module, _inp, output, li=layer_idx):
                acc_fn(C_in, N_in, li, output)
            handles.append(layer.input_layernorm.register_forward_hook(tap_in))

        if hasattr(layer, "post_attention_layernorm"):
            def tap_mlp_in(_module, _inp, output, li=layer_idx):
                acc_fn(C_mlp_in, N_mlp_in, li, output)
            handles.append(layer.post_attention_layernorm.register_forward_hook(tap_mlp_in))

        self_attn = getattr(layer, "self_attn", None)
        if self_attn is not None and hasattr(self_attn, "o_proj"):
            if isinstance(self_attn.o_proj, LowRankLinear):
                def tap_o(module, inputs, _out, li=layer_idx):
                    x = inputs[0]
                    acc_fn(C_attn, N_attn, li, x)
                handles.append(self_attn.o_proj.v_proj.register_forward_hook(tap_o))
            else:
                def tap_o(module, inputs, _out, li=layer_idx):
                    x = inputs[0]
                    acc_fn(C_attn, N_attn, li, x)
                handles.append(self_attn.o_proj.register_forward_hook(tap_o))

        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "down_proj"):
            if isinstance(mlp.down_proj, LowRankLinear):
                def tap_down(module, inputs, _out, li=layer_idx):
                    x = inputs[0]
                    acc_fn(C_h, N_h, li, x)
                handles.append(mlp.down_proj.v_proj.register_forward_hook(tap_down))
            else:
                def tap_down(module, inputs, _out, li=layer_idx):
                    x = inputs[0]
                    acc_fn(C_h, N_h, li, x)
                handles.append(mlp.down_proj.register_forward_hook(tap_down))

    for batch in tqdm(calib_loader, desc="Profiling activations"):
        forward_batch = {
            k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
            if k in ['input_ids', 'attention_mask', 'position_ids', 'labels']
        }
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _ = _model_forward(model, forward_batch, dev)

    for h in handles:
        h.remove()

    def _build_factor(C_dict, N_dict, desc="Building whitening factors"):
        factors = {}
        for layer_idx, gram in tqdm(C_dict.items(), desc=desc, total=len(C_dict)):
            n = max(1, N_dict[layer_idx])
            cov = (gram.float() / n)
            cov = 0.5 * (cov + cov.transpose(0, 1))  # symmetrize

            # Small diagonal jitter for numerical stability
            cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)

            # Replace non finite entries in cov with zeros, then re symmetrize
            if not torch.isfinite(cov).all():
                print(f"[profiling] layer {layer_idx}: non finite entries in cov, zeroing them")
                cov = torch.where(torch.isfinite(cov), cov, torch.zeros_like(cov))
                cov = 0.5 * (cov + cov.transpose(0, 1))
                cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)

            # Try Cholesky once for the easy case
            try:
                R = torch.linalg.cholesky(cov)
                factors[layer_idx] = R.cpu()
                continue
            except (LinAlgError, RuntimeError, Exception):
                pass

            # Robust path, use eigen decomposition only
            try:
                evals, Q = torch.linalg.eigh(cov)
            except (LinAlgError, RuntimeError, Exception):
                print(f"[profiling] layer {layer_idx}: eigendecomposition failed once, cleaning and retrying")
                cov_clean = torch.where(torch.isfinite(cov), cov, torch.zeros_like(cov))
                cov_clean = 0.5 * (cov_clean + cov_clean.transpose(0, 1))
                cov_clean = cov_clean + eps * torch.eye(cov_clean.shape[0], device=cov_clean.device, dtype=cov_clean.dtype)
                evals, Q = torch.linalg.eigh(cov_clean)

            # Clamp eigenvalues, then build a factor without Cholesky
            evals_clamped = torch.clamp(evals, min=eps)
            s = torch.sqrt(evals_clamped)
            R = Q * s.unsqueeze(0)  # R R^T ≈ cov

            # If R still has non finite entries, repair eigenvalues only
            if not torch.isfinite(R).all():
                print(f"[profiling] layer {layer_idx}: non finite R after eigen, repairing")
                mask = torch.isfinite(evals_clamped)
                if not mask.any():
                    evals_clamped = eps * torch.ones_like(evals_clamped)
                else:
                    mean_pos = evals_clamped[mask].mean()
                    evals_clamped = torch.where(mask, evals_clamped, mean_pos)
                s = torch.sqrt(evals_clamped)
                R = Q * s.unsqueeze(0)

            factors[layer_idx] = R.cpu()

        return factors

    L_in = _build_factor(C_in, N_in, desc="Building L_in (Q/K/V whitening)")
    L_attn = _build_factor(C_attn, N_attn, desc="Building L_attn (O_proj whitening)")
    L_mlp_in = _build_factor(C_mlp_in, N_mlp_in, desc="Building L_mlp_in (gate/up whitening)")
    L_h = _build_factor(C_h, N_h, desc="Building L_h (down_proj whitening)")

    profiling_mat = {}
    for layer_idx, layer in enumerate(layers):
        layer_profile = {}
        subset = find_target_modules_in_layer(layer)
        for name, mod in subset.items():
            if any(k in name for k in ["q_proj", "k_proj", "v_proj"]):
                R = L_in.get(layer_idx)
            elif "o_proj" in name:
                R = L_attn.get(layer_idx)
            elif any(k in name for k in ["gate_proj", "up_proj"]):
                R = L_mlp_in.get(layer_idx)
            elif "down_proj" in name:
                R = L_h.get(layer_idx)
            else:
                R = None

            # Get the correct input dimension based on module type
            if isinstance(mod, LowRankLinear):
                dim = mod.in_features
            elif isinstance(mod, nn.Linear):
                dim = mod.in_features
            else:
                raise TypeError(f"Unexpected module type {type(mod)} for {name}")

            if R is None:
                layer_profile[name] = torch.eye(dim, dtype=torch.float32)
            else:
                expected_dim = dim
                if R.shape[0] != expected_dim or R.shape[1] != expected_dim:
                    raise ValueError(
                        f"Profiling matrix dimension mismatch for layer {layer_idx} {name}: "
                        f"expected {expected_dim}x{expected_dim}, got {R.shape}. "
                        f"This indicates a fundamental issue with profiling after compression."
                    )
                layer_profile[name] = R

        profiling_mat[layer_idx] = layer_profile

    model = model.cpu()
    return profiling_mat


@torch.no_grad()
def profle_svdllm_independent(
    model_name: str,
    model,
    calib_loader,
    dev,
    eps: float = 1e-4,
    efficient_accumulate: bool = False,
    chunk_tokens: int = 256,
    chunk_dtype: torch.dtype = torch.bfloat16,
) -> Dict:
    """
    Compute independent input whitening matrices for each target module.

    Unlike profle_svdllm_shared which shares whitening matrices across module types,
    this function computes a separate whitening matrix for each target module key.

    Args:
        efficient_accumulate: If True, use chunked gram accumulation with CPU offload.
        chunk_tokens: Token-chunk size for efficient accumulation.
        chunk_dtype: Dtype used on GPU for chunked gram matmul.
    """
    layers = get_layers(model_name, model)
    model = model.to(dev).eval()
    print("Start obtaining independent input whitening matrices...")
    if efficient_accumulate:
        print(f"   Using efficient accumulation (chunk_tokens={chunk_tokens}, dtype={chunk_dtype})")

    C: Dict[str, torch.Tensor] = {}
    N: Dict[str, int] = {}
    handles = []

    def _accumulate(key: str, tensor: torch.Tensor):
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        B, T, D = tensor.shape
        flat = tensor.reshape(B * T, D).to(dev, dtype=torch.float32)
        gram = flat.transpose(0, 1) @ flat
        if key in C:
            C[key].add_(gram)
            N[key] += B * T
        else:
            C[key] = gram
            N[key] = B * T

    def _accumulate_efficient(key: str, tensor: torch.Tensor):
        """Chunked gram accumulation with CPU offload to reduce GPU memory."""
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        B, T, D = tensor.shape

        if key not in C:
            C[key] = torch.zeros((D, D), device="cpu", dtype=torch.float32)
            N[key] = 0

        flat = tensor.reshape(B * T, D).to(dev, dtype=chunk_dtype)
        for chunk in flat.split(chunk_tokens, dim=0):
            gram_chunk = (chunk.transpose(0, 1) @ chunk).float().cpu()
            C[key].add_(gram_chunk)

        N[key] += B * T

    # Select accumulator function
    acc_fn = _accumulate_efficient if efficient_accumulate else _accumulate

    # Register one hook per target module key
    for layer_idx, layer in enumerate(layers):
        subset = find_target_modules_in_layer(layer)
        for name, mod in subset.items():
            key = f"layer{layer_idx}.{name}"

            if isinstance(mod, LowRankLinear):
                def hook_fn(_m, inputs, _out, k=key):
                    x = inputs[0]
                    acc_fn(k, x)
                handles.append(mod.v_proj.register_forward_hook(hook_fn))
            elif isinstance(mod, nn.Linear):
                def hook_fn(_m, inputs, _out, k=key):
                    x = inputs[0]
                    acc_fn(k, x)
                handles.append(mod.register_forward_hook(hook_fn))

    for batch in tqdm(calib_loader, desc="Profiling activations"):
        forward_batch = {
            k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask", "position_ids", "labels"]
        }
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _ = _model_forward(model, forward_batch, dev)

    for h in handles:
        h.remove()

    # Build one whitening factor per module key
    factors: Dict[str, torch.Tensor] = {}
    for key, gram in tqdm(C.items(), desc="Building whitening factors", total=len(C)):
        n = max(1, N[key])
        cov = (gram.float() / n)
        cov = 0.5 * (cov + cov.transpose(0, 1))
        cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)

        if not torch.isfinite(cov).all():
            cov = torch.where(torch.isfinite(cov), cov, torch.zeros_like(cov))
            cov = 0.5 * (cov + cov.transpose(0, 1))
            cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)

        try:
            R = torch.linalg.cholesky(cov)
        except Exception:
            evals, Q = torch.linalg.eigh(cov)
            evals = torch.clamp(evals, min=eps)
            R = Q * torch.sqrt(evals).unsqueeze(0)

        factors[key] = R.cpu()

    # Repackage into profiling_mat[layer_idx][name] so the rest of your code stays unchanged
    profiling_mat: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer_idx, layer in enumerate(layers):
        layer_profile = {}
        subset = find_target_modules_in_layer(layer)
        for name, mod in subset.items():
            key = f"layer{layer_idx}.{name}"

            if isinstance(mod, LowRankLinear):
                dim = mod.in_features
            elif isinstance(mod, nn.Linear):
                dim = mod.in_features
            else:
                continue

            R = factors.get(key, None)
            if R is None:
                layer_profile[name] = torch.eye(dim, dtype=torch.float32)
            else:
                if R.shape[0] != dim or R.shape[1] != dim:
                    raise ValueError(
                        f"Profiling matrix dimension mismatch for {key}: expected {dim}x{dim}, got {R.shape}"
                    )
                layer_profile[name] = R

        profiling_mat[layer_idx] = layer_profile

    model = model.cpu()
    return profiling_mat


def build_L_cache(
    profiling_mat: Optional[Dict],
    dev: torch.device,
    dtype: torch.dtype = torch.float32
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict]:
    """
    Build CPU cache of whitening matrices from profiling results.

    Returns:
        Tuple of (L_cache, empty dict for backwards compatibility)
    """
    if profiling_mat is None:
        return {}, {}

    L_cache: Dict[int, Dict[str, torch.Tensor]] = {}
    if isinstance(profiling_mat, dict):
        layer_iter = profiling_mat.items()
    else:
        layer_iter = enumerate(profiling_mat)

    for layer_idx, layer_entry in layer_iter:
        L_cache[layer_idx] = {}
        if isinstance(layer_entry, dict):
            items = layer_entry.items()
        else:
            items = []
        for name, L_cpu in items:
            L_cache[layer_idx][name] = L_cpu.to("cpu", dtype=dtype).contiguous()

    return L_cache, {}
