"""
Truncation functions for SVD-LLM compression.

Contains functions for applying SVD truncations to model modules.
"""

import math
from typing import Dict, Set, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.model_utils import get_layers, find_target_modules_in_layer
from utils.correction_utils import (
    get_module_from_key,
    get_module_current_rank,
    factorize_linear,
    replace_module,
    densify_target_modules,
)


def apply_truncations(
    args,
    model,
    tokenizer,
    states,
    planned_ranks: Dict[str, int],
    stage_counts: Dict[str, int],
    reverted_keys: Set[str],
    L_cache,
    svd_cache,
    teacher_weights: Dict[str, torch.Tensor],
    teacher_biases: Dict[str, torch.Tensor],
    dev,
    substituted_keys: Optional[Set[str]] = None,
):
    """
    Apply truncations to modules based on planned_ranks.

    Args:
        args: Compression arguments
        model: The language model
        tokenizer: Tokenizer (unused but kept for API compatibility)
        states: Dict of ModulePruneState objects
        planned_ranks: Dict of target ranks per module
        stage_counts: Dict of counts per module for this stage
        reverted_keys: Set of module keys that were reverted
        L_cache: Cached whitening matrices
        svd_cache: Cached SVD results
        teacher_weights: Dict of teacher weight tensors
        teacher_biases: Dict of teacher bias tensors
        dev: Device to use
        substituted_keys: Mutable set to track substituted modules

    Returns:
        Set of newly substituted module keys
    """
    if substituted_keys is None:
        substituted_keys = set()

    newly_substituted = set()
    sorted_stage_items = sorted(stage_counts.items(), key=lambda kv: kv[0])
    trunc_pbar = tqdm(sorted_stage_items, desc="Applying truncations", leave=True)

    for key, _ in trunc_pbar:
        if key in reverted_keys:
            continue

        module, layer_idx, module_name = get_module_from_key(args.model, model, key)
        cached_L = L_cache.get(layer_idx, {}).get(module_name) if L_cache else None
        if cached_L is None:
            L = torch.eye(module.in_features, device=dev, dtype=torch.float32)
        else:
            L = cached_L.to(dev, dtype=torch.float32)

        state = states[key]
        target_rank = planned_ranks[key]

        # Condition for compression
        should_compress = target_rank <= state.r_star

        if should_compress:
            svd_info = svd_cache.get(key, None) if svd_cache is not None else None

            low_rank = factorize_linear(
                module,
                L,
                target_rank,
                dev,
                use_triangular_solve=args.use_triangular_solve,
                keep_mask=state.keep_mask,
                svd_info=svd_info,
                mask_rank_only=state.mask_rank_only,
                remap_fake_bnb=getattr(args, "remap", False),
                quantize_8_bit=getattr(args, "quantize_8_bit", False),
            )
            replace_module(args.model, model, key, low_rank.to(dev))
        else:
            # No benefit or no planned change
            if getattr(args, "sub_with_teacher_module", False) and target_rank > state.r_star:
                w_teacher = teacher_weights.get(key, None)
                if w_teacher is not None:
                    with torch.no_grad():
                        module.weight.data.copy_(
                            w_teacher.to(module.weight.device, dtype=module.weight.dtype)
                        )
                        b_teacher = teacher_biases.get(key, None)
                        if b_teacher is not None and getattr(module, "bias", None) is not None:
                            module.bias.data.copy_(
                                b_teacher.to(module.bias.device, dtype=module.bias.dtype)
                            )
                    substituted_keys.add(key)
                    newly_substituted.add(key)

        states[key] = state

    trunc_pbar.close()
    return newly_substituted


def _vanilla_target_rank(m: int, n: int, pruning_ratio: float) -> int:
    """
    Compute rank r so that r*(m+n) ≈ (1-pruning_ratio)*m*n.
    r = (1-pruning_ratio) * (m*n)/(m+n), clamped to [1, min(m,n)].

    Args:
        m: Output dimension
        n: Input dimension
        pruning_ratio: Fraction of parameters to remove

    Returns:
        Target rank
    """
    full_rank = min(m, n)
    denom = max(1.0, float(m + n))
    r = int(math.floor((1.0 - float(pruning_ratio)) * (float(m) * float(n)) / denom))
    r = max(1, min(r, full_rank))
    return r


def run_vanilla_svd_stage(
    args,
    model,
    tokenizer,
    dev,
    stage_idx: int,
):
    """
    One outer stage of plain SVD truncation on each target module weight W.
    Uses L = I, so factorize_linear runs SVD directly in the unwhitened space.

    Args:
        args: Compression arguments
        model: The language model
        tokenizer: Tokenizer (unused but kept for API compatibility)
        dev: Device to use
        stage_idx: Current stage index

    Returns:
        Updated model
    """
    print(f"\n⚙️  Vanilla SVD stage {stage_idx}: truncating target modules in raw SVD space")

    # Densify first so every module has a single weight matrix W we can truncate again.
    densified_map = densify_target_modules(args.model, model, verbose=args.verbose)
    prev_ranks = {}
    for k, old_mod in densified_map.items():
        # old_mod is the pre-densify module, often LowRankLinear
        try:
            r_old = get_module_current_rank(old_mod)
            if r_old is not None:
                prev_ranks[str(k)] = int(r_old)
        except Exception:
            pass

    layers = get_layers(args.model, model)
    model = model.to(dev).eval()

    num_factorized = 0
    for layer_idx, layer in enumerate(tqdm(layers, desc=f"Vanilla SVD stage {stage_idx}", total=len(layers))):
        subset = find_target_modules_in_layer(layer)
        for name, mod in subset.items():
            # After densify, target modules should be nn.Linear here
            if not isinstance(mod, nn.Linear):
                continue

            key = f"layer{layer_idx}.{name}"
            W = mod.weight
            m, n = int(W.shape[0]), int(W.shape[1])
            full_rank = min(m, n)

            r_target = _vanilla_target_rank(m, n, args.global_prune_ratio)

            # Never increase rank if this module was already low rank before densify
            r_prev = prev_ranks.get(key, full_rank)
            r_target = min(r_target, r_prev)

            # If module was previously low rank, we should re-factorize back even if r_target == r_prev
            should_factorize = (r_prev < full_rank) or (r_target < full_rank)
            if not should_factorize:
                continue

            L = torch.eye(n, device=dev, dtype=torch.float32)
            low_rank = factorize_linear(
                mod,
                L,
                r_target,
                dev,
                use_triangular_solve=False,
                keep_mask=None,
                svd_info=None,
                mask_rank_only=False,
                remap_fake_bnb=getattr(args, "remap", False),
                quantize_8_bit=getattr(args, "quantize_8_bit", False),
            )
            replace_module(args.model, model, key, low_rank.to(dev))
            num_factorized += 1

    # Free densified modules to release memory
    for old_mod in densified_map.values():
        for p in old_mod.parameters():
            p.data = p.data.cpu()
    del densified_map
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    print(f"   ✅ Vanilla SVD stage {stage_idx}: factorized {num_factorized} modules")
    return model
