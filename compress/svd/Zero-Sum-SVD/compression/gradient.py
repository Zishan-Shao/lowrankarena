"""
Gradient computation and importance recomputation for SVD-LLM compression.

Contains the main gradient accumulation and importance orchestration functions
extracted from main_zero_sum.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

from utils.model_utils import get_layers, find_target_modules_in_layer
from utils.correction_utils import (
    densify_target_modules,
    restore_target_modules,
)
from compression.profiling import (
    svd_whitened_sorted,
    prepare_calib_loader,
    compute_profiling_matrices,
    build_L_cache,
)
from compression.importance import (
    _importance_autocast_dtype,
    compute_importance_for_layer_fast,
    compute_grad_sigma_efficient_for_module,
)
from compression.batch_utils import _model_forward


def compute_average_gradients(model, tokenizer, cali_data, dev, layers, args=None):
    """
    Compute average gradients over calibration data for all target modules.

    Args:
        model: The language model
        tokenizer: Tokenizer (unused but kept for API compatibility)
        cali_data: Calibration data batches
        dev: Device to use
        layers: Model layers
        args: Arguments with options for loss type, sample selection, etc.

    Returns:
        Dict mapping layer_idx -> module_name -> average gradient tensor
    """
    print("\nComputing average gradients over calibration data...")
    model.train()
    model.config.use_cache = False

    # Enable gradient checkpointing to reduce activation memory during importance gradient pass
    # This trades compute for much lower activation memory with long sequences (T=2048)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("   ✓ Gradient checkpointing enabled (non-reentrant)")
    except TypeError:
        # Fallback for older transformers versions that don't support gradient_checkpointing_kwargs
        model.gradient_checkpointing_enable()
        print("   ✓ Gradient checkpointing enabled")

    target_params: List[torch.Tensor] = []
    target_keys: List[Tuple[int, str]] = []
    grad_sums: Dict[int, Dict[str, torch.Tensor]] = {}

    for i, layer in enumerate(layers):
        grad_sums[i] = {}
        for name, mod in find_target_modules_in_layer(layer).items():
            if isinstance(mod, nn.Linear):
                target_params.append(mod.weight)
                target_keys.append((i, name))
                # aggregate in fp32 on CPU to reduce GPU memory usage
                grad_sums[i][name] = torch.zeros_like(mod.weight, device="cpu", dtype=torch.float32)

    if not target_params:
        return grad_sums

    autocast_dtype = (
        torch.bfloat16
        if any(p.dtype == torch.bfloat16 for p in model.parameters())
        else torch.float16
    )

    print("Using standard CE loss for importance gradient computation")

    # New flag to control how many calibration samples are used for gradients
    max_grad_samples = None
    if args is not None and getattr(args, "nsamples_gradient_subset", None) is not None:
        try:
            max_grad_samples = int(args.nsamples_gradient_subset)
        except Exception:
            max_grad_samples = None
        if max_grad_samples is not None and max_grad_samples <= 0:
            max_grad_samples = None

    if max_grad_samples is not None:
        print(f"Using at most {max_grad_samples} calibration samples for gradients")

    global_sample_ptr = 0
    batch_count = 0

    # Set progress bar total to the actual number of samples we'll process
    total_samples = len(cali_data)
    if max_grad_samples is not None:
        total_samples = min(len(cali_data), max_grad_samples)

    for batch in tqdm(cali_data, desc="Computing gradients", total=total_samples):
        # Stop if we already used the requested number of samples
        if max_grad_samples is not None and global_sample_ptr >= max_grad_samples:
            break

        # Move batch to device, but keep full tensors so we can slice by B_use
        raw_ids = batch["input_ids"].to(dev)
        raw_am = batch.get("attention_mask", None)
        if raw_am is not None:
            raw_am = raw_am.to(dev)
        else:
            raw_am = torch.ones_like(raw_ids)

        raw_pi = batch.get("position_ids", None)
        if raw_pi is not None:
            raw_pi = raw_pi.to(dev)

        B_all = raw_ids.size(0)

        # Decide how many samples from this batch to actually use
        if max_grad_samples is not None:
            remaining = max_grad_samples - global_sample_ptr
            if remaining <= 0:
                break
            B_use = min(B_all, remaining)
        else:
            B_use = B_all

        if B_use <= 0:
            break

        ids = raw_ids[:B_use]
        am = raw_am[:B_use]
        position_ids = raw_pi[:B_use] if raw_pi is not None else None
        B = ids.size(0)

        # Trim to the maximum number of non-pad tokens in this batch
        actual_len = int(am.sum(dim=1).max().item())
        actual_len = max(actual_len, 2)  # need >=2 tokens for shift
        ids = ids[:, :actual_len]
        am = am[:, :actual_len]
        if position_ids is not None:
            position_ids = position_ids[:, :actual_len]

        # Optional truncation just for importance gradient computation
        # controlled by args.importance_seq_len and a guard flag
        max_len = None
        if args is not None and getattr(args, "_in_importance_pass", False):
            max_len = getattr(args, "importance_seq_len", None)

        if max_len is not None and max_len > 0 and ids.size(1) > max_len:
            ids = ids[:, -max_len:]
            am = am[:, -max_len:]
            if position_ids is not None:
                position_ids = position_ids[:, -max_len:]

        forward_batch = {
            "input_ids": ids,
            "attention_mask": am,
        }
        if position_ids is not None:
            forward_batch["position_ids"] = position_ids

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            out = _model_forward(model, forward_batch, dev)
            logits = out.logits

            # Compute CE loss from existing logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = ids[..., 1:].contiguous()

            per_tok = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(ids.size(0), -1)  # [B, T_minus_1]

            valid = am[..., 1:].contiguous().bool()

            tok_mask = batch.get("token_mask", None)
            if tok_mask is not None:
                tok_mask = tok_mask.to(dev).bool()
                mask = valid & tok_mask
                if mask.any():
                    loss = per_tok[mask].mean()
                else:
                    # If the selected token became invalid after truncation, skip this micro batch
                    del out, logits, per_tok
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
            else:
                loss = per_tok[valid].mean()

        grads = torch.autograd.grad(
            loss,
            target_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        for (layer_idx, module_name), grad_tensor in zip(target_keys, grads):
            if grad_tensor is None:
                continue
            # move to CPU and cast to fp32 before adding
            grad_sums[layer_idx][module_name].add_(grad_tensor.detach().to("cpu", dtype=torch.float32))

        global_sample_ptr += B
        batch_count += 1

        del out, logits, grads
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # Average by the number of gradient batches actually used, not by len(cali_data)
    n = max(batch_count, 1)
    for layer_idx in grad_sums:
        for name in grad_sums[layer_idx]:
            grad_sums[layer_idx][name].div_(n)

    # Disable gradient checkpointing to restore model state
    model.gradient_checkpointing_disable()

    return grad_sums


def recompute_importance(args, model, tokenizer, dev, existing_profiling=None):
    """
    Recompute profiling and singular value importance for the current model.

    Key design:
      * Profiling is computed on the real model (with LowRankLinear where present).
      * For gradients and importance we temporarily densify each target LowRankLinear
        into a single nn.Linear with weight W = U @ V.
      * Importance is always keyed per logical module (q_proj, k_proj, v_proj, o_proj,
        gate_proj, up_proj, down_proj), never separately for u_proj and v_proj.

    Args:
        args: Compression arguments
        model: The language model
        tokenizer: Tokenizer for calibration data
        dev: Device to use
        existing_profiling: Optional pre-computed profiling matrices (unused)

    Returns:
        Tuple of (importance, profiling_mat, L_cache, svd_cache)
    """
    # 1) Build base calibration data once
    layers = get_layers(args.model, model)
    base_cali_data = prepare_calib_loader(args, tokenizer)  # fixed set of length nsamples

    # Debug: print first sample used for importance computation
    if base_cali_data:
        import hashlib
        first_sample = base_cali_data[0]["input_ids"]
        print(f"🔍 Importance computation - first sample token IDs[:20]: {first_sample.flatten()[:20].tolist()}")
        calib_hash = hashlib.md5(
            torch.cat([b["input_ids"].flatten() for b in base_cali_data]).numpy().tobytes()
        ).hexdigest()
        print(f"🔍 Importance computation - calibration data hash: {calib_hash}")

    # 2) Profiling matrices for current compressed model (before densification)
    print("\n🔄 Recomputing profiling matrices for updated model structure...")
    if getattr(args, "selection_mode", None) == "delta_in_subspace":
        print("   Using delta_in_subspace selection; singular values are scored")
        print("   by projection of their induced change in W onto avg gradient")
    profiling_mat = compute_profiling_matrices(args, model, tokenizer, dev)
    print("   ✓ Profiling matrices recomputed\n")

    L_cache, _ = build_L_cache(profiling_mat, dev, dtype=torch.float32)

    # 2b) Build calibration data for gradients
    grad_cali_data = base_cali_data

    # 3) Densify target LowRankLinear modules into single nn.Linear with W = U @ V
    verbose = getattr(args, 'verbose', False)
    replaced = densify_target_modules(args.model, model, verbose=verbose)
    if replaced and not verbose:
        print(f"   📦 Densified {len(replaced)} LowRankLinear modules for gradient computation")

    # 4) Gradient pass on densified model, using possibly random subset
    model = model.to(dev)

    # Mark this call as the importance pass so sequence truncation can apply
    prev_flag = getattr(args, "_in_importance_pass", False)
    setattr(args, "_in_importance_pass", True)

    # Build list of all modules for per-module progress bar
    all_modules = []
    for layer_idx, layer in enumerate(layers):
        subset = find_target_modules_in_layer(layer)
        for name, mod in subset.items():
            if isinstance(mod, nn.Linear):
                all_modules.append((layer_idx, name, mod))

    try:
        if getattr(args, "efficient_importance", False):
            # ===== EFFICIENT IMPORTANCE MODE =====
            if getattr(args, "selection_mode", None) == "delta_in_subspace":
                raise ValueError("efficient_importance does not support selection_mode=delta_in_subspace")

            print("   ✅ Using efficient_importance mode, per module grad_sigma accumulation")

            model.train()
            model.config.use_cache = False

            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()

            autocast_dtype = _importance_autocast_dtype(model)

            efficient_grad_cali_data = grad_cali_data

            importance: Dict[str, Dict] = {}
            svd_cache: Dict[str, Dict[str, torch.Tensor]] = {}

            for layer_idx, name, mod in tqdm(all_modules, desc="Recomputing importance (efficient)"):
                key = f"layer{layer_idx}.{name}"

                W = mod.weight.detach().to(dev, dtype=torch.float32)
                shape = tuple(W.shape)

                L = L_cache.get(layer_idx, {}).get(name, None)
                if L is None:
                    dim = W.shape[1]
                    L = torch.eye(dim, device=dev, dtype=torch.float32)
                else:
                    L = L.to(dev, dtype=torch.float32)

                U_svd, S, Vh_svd = svd_whitened_sorted(W, L)
                V = Vh_svd.transpose(0, 1)

                grad_sigma_cpu = compute_grad_sigma_efficient_for_module(
                    model=model,
                    cali_data=efficient_grad_cali_data,
                    dev=dev,
                    mod=mod,
                    U=U_svd,
                    V=V,
                    L=L,
                    args=args,
                    autocast_dtype=autocast_dtype,
                )

                grad_sigma = grad_sigma_cpu.to(dev)
                saliency = grad_sigma.abs()

                importance[key] = {
                    "sigma": S.detach().cpu(),
                    "grad_sigma": grad_sigma.detach().cpu(),
                    "saliency": saliency.detach().cpu(),
                    "shape": shape,
                }

                svd_cache[key] = {
                    "S": S.detach().cpu(),
                    "U": U_svd.detach().cpu(),
                    "Vh": Vh_svd.detach().cpu(),
                }

                del W, L, U_svd, Vh_svd, V, grad_sigma, saliency
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

            model.gradient_checkpointing_disable()
            model.eval()

        else:
            # ===== ORIGINAL (NON-EFFICIENT) IMPORTANCE MODE =====
            avg_grads = compute_average_gradients(
                model,
                tokenizer,
                grad_cali_data,  # random subset or full base set
                dev,
                layers,
                args=args,
            )

            # 5) Compute importance per logical module using W and G and L
            importance: Dict[str, Dict] = {}
            svd_cache: Dict[str, Dict[str, torch.Tensor]] = {}

            with torch.no_grad():
                for layer_idx, name, mod in tqdm(all_modules, desc="Recomputing importance"):
                    key = f"layer{layer_idx}.{name}"

                    # Dense weight W (this already equals U @ V if the module was low rank)
                    W = mod.weight.detach().to(dev, dtype=torch.float32)
                    shape = tuple(W.shape)

                    # Average gradient for this logical W
                    layer_grads = avg_grads.get(layer_idx, {})
                    G = layer_grads.get(name, None)
                    if G is None:
                        G = torch.zeros_like(W, device=dev, dtype=torch.float32)
                    else:
                        G = G.to(dev, dtype=torch.float32)

                    # Whitening matrix for this module
                    L = L_cache.get(layer_idx, {}).get(name, None)
                    if L is None:
                        dim = W.shape[1]
                        L = torch.eye(dim, device=dev, dtype=torch.float32)
                    else:
                        L = L.to(dev, dtype=torch.float32)

                    # Singular values and importance for W @ L; also get U and Vh
                    S, grad_sigma, saliency, U_svd, Vh_svd = compute_importance_for_layer_fast(
                        W, G, L, use_triangular_solve=args.use_triangular_solve
                    )

                    # delta_in_subspace selection uses projection of the unwhitened
                    # singular value drop onto the average gradient direction of W
                    if getattr(args, "selection_mode", None) == "delta_in_subspace":
                        # Flatten average gradient as subspace direction
                        g_flat = G.view(-1)
                        g_norm = g_flat.norm()

                        if g_norm > 0:
                            # Build L inverse once per module for ΔW computation
                            try:
                                L_inv = torch.linalg.inv(L)
                            except Exception:
                                L_reg = L + 1e-6 * torch.eye(
                                    L.shape[0], device=L.device, dtype=L.dtype
                                )
                                L_inv = torch.linalg.inv(L_reg)

                            # Vectorized computation of projection scores
                            # score_j = |S[j] * u_j^T (G @ L_inv) v_j|
                            V = Vh_svd.transpose(0, 1)              # [n, r]
                            Gtilde = G @ L_inv                      # [m, n]
                            Ug = U_svd.transpose(0, 1) @ Gtilde     # [r, n]
                            UGV = Ug @ V                            # [r, r]

                            # Inner products u_j^T Gtilde v_j
                            inner = torch.diagonal(UGV, dim1=-2, dim2=-1)  # [r]
                            scores = (S * inner).abs()

                            # Encode scores into grad_sigma so that delta = -sigma * grad_sigma = scores
                            eps_sigma = 1e-12
                            grad_sigma = -scores / (S.abs() + eps_sigma)
                            saliency = scores
                        else:
                            # If gradient is zero, fall back to magnitude based importance
                            saliency = S.abs()
                            grad_sigma = torch.ones_like(S)

                    importance[key] = {
                        "sigma": S.detach().cpu(),
                        "grad_sigma": grad_sigma.detach().cpu(),
                        "saliency": saliency.detach().cpu(),
                        "shape": shape,
                    }

                    svd_cache[key] = {
                        "S": S.detach().cpu(),
                        "U": U_svd.detach().cpu(),
                        "Vh": Vh_svd.detach().cpu(),
                    }

                    del W, G

                    if dev.type == "cuda":
                        torch.cuda.empty_cache()

    finally:
        setattr(args, "_in_importance_pass", prev_flag)

    # 6) Restore original LowRankLinear modules back into the model
    if replaced:
        restore_target_modules(args.model, model, replaced)
        if not verbose:
            print(f"   ✅ Restored {len(replaced)} LowRankLinear modules")

    # Debug: Print the keys we're using in importance
    print("\n   📊 Importance keys after recomputation:")
    sample_keys = list(importance.keys())[:5]  # First 5 keys
    for k in sample_keys:
        print(f"      - {k} (shape: {importance[k]['shape']})")
    if len(importance) > 5:
        print(f"      ... and {len(importance) - 5} more")

    # Check for any .u_proj or inner .v_proj keys (should be none)
    # Note: self_attn.v_proj is legitimate (value projection)
    # We want to catch things like self_attn.v_proj.v_proj or self_attn.q_proj.u_proj
    bad_keys = []
    for k in importance.keys():
        # Check for .u_proj anywhere (always bad since we don't have a "u" projection)
        if '.u_proj' in k:
            bad_keys.append(k)
        # Check for nested .v_proj (e.g., v_proj.v_proj)
        elif k.count('.v_proj') > 1:
            bad_keys.append(k)

    if bad_keys:
        print(f"   ❌ WARNING: Found {len(bad_keys)} bad keys with .u_proj or nested .v_proj:")
        for k in bad_keys[:3]:
            print(f"      - {k}")
    else:
        print("   ✓ All keys are logical module keys (no .u_proj or nested .v_proj)")

    return importance, profiling_mat, L_cache, svd_cache


def get_or_recompute_importance(args, model, tokenizer, dev, stage_idx,
                                 cached_importance, cached_profiling, cached_svd_cache):
    """
    Return (importance, profiling_mat, L_cache, svd_cache).
    Reuses cached values for stage_idx=1 if available.

    Args:
        args: Compression arguments
        model: The language model
        tokenizer: Tokenizer
        dev: Device
        stage_idx: Current stage index (1-based)
        cached_importance: Previously cached importance dict or None
        cached_profiling: Previously cached profiling matrices or None
        cached_svd_cache: Previously cached SVD cache or None

    Returns:
        Tuple of (importance, profiling_mat, L_cache, svd_cache)
    """
    if stage_idx == 1 and cached_importance is not None:
        print(f"\n📋 Using precomputed importance for stage {stage_idx}")
        return cached_importance, cached_profiling, None, cached_svd_cache
    else:
        if stage_idx == 1:
            print(f"\n🔁 Computing importance for first time (stage {stage_idx})...")
        else:
            print(f"\n🔁 Recomputing importance for stage {stage_idx}...")
        return recompute_importance(args, model, tokenizer, dev)
