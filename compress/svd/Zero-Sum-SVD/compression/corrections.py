"""
Between-stage correction functions for SVD-LLM compression.

Contains functions for applying corrections between compression stages
including subspace projection, delta projection, and alpha blending.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import LinAlgError
from typing import Dict, List, Set, Optional
from tqdm import tqdm

from utils.model_utils import get_layers, find_target_modules_in_layer
from utils.correction_utils import (
    extract_dense_weight,
    densify_target_modules,
    run_alpha_blend_between_stages,
)
from compression.gradient import compute_average_gradients
from compression.batch_utils import _model_forward
from utils.eval_utils import ppl_eval


def run_pull_subspace_between_stages(
    args,
    model,
    tokenizer,
    dev,
    teacher_weights: Dict[str, torch.Tensor],
    stage_idx: int,
    substituted_keys: Optional[Set[str]] = None,
):
    """
    Subspace based projection between stages.

    For each target module:
      * w_pre is the dense teacher weight from the uncompressed model
      * w_r is the current truncated dense weight after densification
      * delta = w_pre - w_r  on CPU
      * For nsamples_subspace_proj calibration batches, compute gradients
        g_1,...,g_N for that module at w_r
      * Build an orthonormal subspace from these gradients and project
        delta onto span{g_i} in a numerically exact way, using

            projection(delta) = G (G^T G)^{-1} G^T delta,

        where G has rows g_i flattened

      To keep memory under control, we process modules one by one, so at
      most N gradient vectors for a single module live in memory at once.
      Model weights remain fixed until all module corrections are computed,
      so all gradients are taken at w_r.

    Args:
        args: Compression arguments
        model: The language model
        tokenizer: Tokenizer (unused but kept for API compatibility)
        dev: Device to use
        teacher_weights: Dict of teacher weight tensors
        stage_idx: Current stage index
        substituted_keys: Optional set of module keys that were substituted
    """
    if substituted_keys is None:
        substituted_keys = set()

    if not getattr(args, "pull_subspace", False):
        return

    if not hasattr(args, "base_calib_data") or args.base_calib_data is None:
        print("   Pull subspace requested but base calibration data is missing, skipping for this stage")
        return

    print(f"\n🧲 Stage {stage_idx}: starting pull subspace correction with CE (cross entropy)")

    model.to(dev)

    # Turn off KV cache to avoid blowing up training memory
    prev_use_cache = None
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        prev_use_cache = model.config.use_cache
        model.config.use_cache = False

    model.train()

    base_list = args.base_calib_data
    total_batches = len(base_list)

    # For KD we need the global sample index for each entry in base_list
    # so that we can map to the correct teacher .pt files
    batch_offsets = []
    cum_offset = 0
    for b in base_list:
        B_here = b["input_ids"].size(0)
        batch_offsets.append(cum_offset)
        cum_offset += B_here

    # Determine how many calibration batches to use for the subspace
    ns_proj = getattr(args, "nsamples_subspace_proj", None)
    if ns_proj is None or ns_proj <= 0:
        ns_proj = total_batches
    ns_proj = min(ns_proj, total_batches)

    # Decide whether to randomize batch indices for pull_subspace
    random_subspace = getattr(args, "random_samples_for_pullsubspace", False)

    if random_subspace:
        # Random selection of ns_proj distinct batch indices
        batch_indices = torch.randperm(total_batches)[:ns_proj].tolist()
    else:
        # Original behavior, use the first ns_proj batches
        batch_indices = list(range(ns_proj))

    print(f"   Using {len(batch_indices)} calibration batches for subspace gradients "
          f"(out of {total_batches})")

    # Pre compute delta weights on CPU for all target modules at this stage
    # delta[key] = (w_teacher - w_truncated) in fp32 on CPU
    delta_weights: Dict[str, torch.Tensor] = {}
    layers = get_layers(args.model, model)
    for layer_idx, layer in enumerate(layers):
        subset = find_target_modules_in_layer(layer)
        for name, mod in subset.items():
            key = f"layer{layer_idx}.{name}"

            # If this module was substituted with the teacher after truncation,
            # skip subspace projection correction
            if key in substituted_keys:
                continue

            w_teacher = teacher_weights.get(key, None)
            if w_teacher is None:
                continue
            w_r = extract_dense_weight(mod).detach().cpu()
            if w_teacher.shape != w_r.shape:
                continue
            # Store delta in fp32 on CPU
            delta_weights[key] = (w_teacher.to(dtype=torch.float32) - w_r.to(dtype=torch.float32))

    if not delta_weights:
        print("   No matching teacher weights found for subspace projection, skipping")
        model.eval()
        return

    # Autocast dtype based on model parameters
    autocast_dtype = (
        torch.bfloat16
        if any(p.dtype == torch.bfloat16 for p in model.parameters())
        else torch.float16
    )

    corrections: Dict[str, torch.Tensor] = {}
    eps = 1e-6
    num_modules_processed = 0
    num_modules_skipped = 0

    # Build list of modules to process for progress bar
    modules_to_process = []
    for layer_idx, layer in enumerate(layers):
        subset = find_target_modules_in_layer(layer)
        for name, mod in subset.items():
            if not isinstance(mod, nn.Linear):
                continue
            key = f"layer{layer_idx}.{name}"
            if key in delta_weights:
                modules_to_process.append((layer_idx, name, mod, key))

    pbar = tqdm(modules_to_process, desc=f"Pull subspace stage {stage_idx}", unit="module")
    for layer_idx, name, mod, key in pbar:
        delta_cpu = delta_weights[key]
        pbar.set_postfix({"module": f"L{layer_idx}.{name}"})

        delta_flat = delta_cpu.view(-1)
        grad_rows: List[torch.Tensor] = []

        for batch_idx in batch_indices:
            batch = base_list[batch_idx]
            raw_ids = batch["input_ids"].to(dev)
            raw_am = batch.get("attention_mask", None)
            if raw_am is not None:
                raw_am = raw_am.to(dev)
            else:
                raw_am = torch.ones_like(raw_ids)

            raw_pi = batch.get("position_ids", None)
            if raw_pi is not None:
                raw_pi = raw_pi.to(dev)

            ids = raw_ids
            am = raw_am
            position_ids = raw_pi

            # Trim to the maximum number of non-pad tokens in this batch
            actual_len = int(am.sum(dim=1).max().item())
            actual_len = max(actual_len, 2)  # need >=2 tokens for shift
            ids = ids[:, :actual_len]
            am = am[:, :actual_len]
            if position_ids is not None:
                position_ids = position_ids[:, :actual_len]

            # Optionally truncate sequence length to save memory
            max_len_proj = getattr(args, "subspace_proj_seq_len", 1024)
            if ids.size(1) > max_len_proj:
                ids = ids[:, -max_len_proj:]
                am = am[:, -max_len_proj:]
                if position_ids is not None:
                    position_ids = position_ids[:, -max_len_proj:]

            B = ids.size(0)

            forward_batch = {
                "input_ids": ids,
                "attention_mask": am,
            }
            if position_ids is not None:
                forward_batch["position_ids"] = position_ids

            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                out = _model_forward(model, forward_batch, dev)
                logits = out.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = ids[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='mean'
                )

            grad_tensor = torch.autograd.grad(
                loss,
                mod.weight,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]

            if grad_tensor is not None:
                grad_flat = grad_tensor.detach().to("cpu", dtype=torch.float32).view(-1)
                grad_rows.append(grad_flat)

            # Clean up per step
            del out, logits, loss
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        if not grad_rows:
            num_modules_skipped += 1
            continue

        G = torch.stack(grad_rows, dim=0)  # [N_used, P]
        # Compute G^T G and G^T delta in sample space
        c_vec = torch.mv(G, delta_flat)            # [N_used]
        S_mat = G @ G.t()                          # [N_used, N_used]
        S_mat = S_mat + eps * torch.eye(S_mat.size(0), device=S_mat.device, dtype=S_mat.dtype)

        try:
            alpha = torch.linalg.solve(S_mat, c_vec)  # [N_used]
        except LinAlgError:
            # Fall back to least squares
            alpha, _ = torch.lstsq(c_vec.unsqueeze(1), S_mat)
            alpha = alpha.squeeze(1)

        # projection(delta) = sum_i alpha_i g_i
        proj_flat = torch.matmul(alpha.unsqueeze(0), G).squeeze(0)  # [P]
        proj_tensor = proj_flat.view_as(delta_cpu)

        corrections[key] = proj_tensor
        num_modules_processed += 1

        # Free this module gradients early
        del G, c_vec, S_mat, proj_flat, proj_tensor, grad_rows
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if num_modules_processed == 0:
        print("   No modules received usable gradients for pull subspace, skipping weight update")
        model.eval()
        return

    print(f"   Pull subspace computed corrections for {num_modules_processed} modules, skipped {num_modules_skipped}")

    # Apply all corrections in one shot to keep gradients consistent with w_truncated
    with torch.no_grad():
        for layer_idx, layer in enumerate(layers):
            subset = find_target_modules_in_layer(layer)
            for name, mod in subset.items():
                key = f"layer{layer_idx}.{name}"
                corr_cpu = corrections.get(key, None)
                if corr_cpu is None:
                    continue
                corr = corr_cpu.to(mod.weight.device, dtype=mod.weight.dtype)
                mod.weight.data.add_(corr)

    model.eval()

    print(f"   Pull subspace correction complete on {num_modules_processed} modules\n")

    # Restore cache flag
    if prev_use_cache is not None:
        model.config.use_cache = prev_use_cache


def run_project_to_delta_between_stages(
    args,
    model,
    tokenizer,
    dev,
    teacher_weights: Dict[str, torch.Tensor],
    stage_idx: int,
    substituted_keys: Optional[Set[str]] = None,
):
    """
    Module-wise projection correction between stages.

    For each target module:
      * W_r is current dense weight after densification
      * W_T is cached dense teacher weight
      * delta = W_T - W_r
      * compute average gradient G at W_r over a subset of base calibration data
      * project G onto delta:
            proj = ( <G, delta> / <delta, delta> ) * delta
      * update: W_r <- W_r + proj

    No fixed learning rate is used.

    Args:
        args: Compression arguments
        model: The language model
        tokenizer: Tokenizer
        dev: Device to use
        teacher_weights: Dict of teacher weight tensors
        stage_idx: Current stage index
        substituted_keys: Optional set of module keys that were substituted
    """
    if substituted_keys is None:
        substituted_keys = set()

    if not getattr(args, "project_to_delta", False):
        return

    base_list = getattr(args, "base_calib_data", None)
    if base_list is None:
        print("   project_to_delta requested but base calibration data is missing, skipping")
        return

    # Decide how many calibration entries to use
    ns = getattr(args, "n_samples_proj_to_delta", None)
    if ns is not None:
        try:
            ns = int(ns)
        except Exception:
            ns = None
        if ns is not None and ns <= 0:
            ns = None

    grad_cali_data = base_list if ns is None else base_list[:min(ns, len(base_list))]

    print(f"\n🧭 Stage {stage_idx}: project_to_delta using {len(grad_cali_data)} calibration batches")

    # Build layers list
    layers = get_layers(args.model, model)

    # Compute average gradients on the current model
    # Pass args=None to force standard CE path and avoid coupling to importance flags
    avg_grads = compute_average_gradients(
        model,
        tokenizer,
        grad_cali_data,
        dev,
        layers,
        args=None,
    )

    eps = 1e-12
    num_updated = 0
    num_skipped = 0

    with torch.no_grad():
        for layer_idx, layer in enumerate(layers):
            subset = find_target_modules_in_layer(layer)
            for name, mod in subset.items():
                if not isinstance(mod, nn.Linear):
                    continue

                key = f"layer{layer_idx}.{name}"

                # Respect teacher substitution skip behavior
                if key in substituted_keys:
                    num_skipped += 1
                    continue

                w_teacher = teacher_weights.get(key, None)
                if w_teacher is None:
                    num_skipped += 1
                    continue

                # Current weight
                W_r = mod.weight.data
                if W_r is None:
                    num_skipped += 1
                    continue

                # Shapes must match
                if tuple(w_teacher.shape) != tuple(W_r.shape):
                    num_skipped += 1
                    continue

                # Average gradient tensor for this logical module
                layer_grads = avg_grads.get(layer_idx, {})
                G = layer_grads.get(name, None)
                if G is None:
                    num_skipped += 1
                    continue

                # Move teacher to device for delta computation
                delta = w_teacher.to(dev, dtype=torch.float32) - W_r.detach().to(dev, dtype=torch.float32)

                denom = torch.sum(delta * delta)
                if denom.abs().item() < eps:
                    num_skipped += 1
                    continue

                Gf = G.to(dev, dtype=torch.float32)
                numer = torch.sum(Gf * delta)

                alpha = numer / denom
                proj = alpha * delta  # same shape as W

                # Snap by adding the projected component directly, no fixed lr
                W_r.add_(proj.to(W_r.dtype))

                num_updated += 1

    print(f"   ✅ project_to_delta updated {num_updated} modules, skipped {num_skipped}\n")


def run_between_stage_corrections(
    args,
    model,
    tokenizer,
    dev,
    teacher_weights: Dict[str, torch.Tensor],
    stage_idx: int,
    is_final_stage: bool,
    substituted_keys: Optional[Set[str]] = None,
):
    """
    Densify modules and optionally run alpha_blend, project_to_delta, and pull_subspace.
    For final stage, just prints a message and returns.

    Args:
        args: Compression arguments
        model: The language model
        tokenizer: Tokenizer
        dev: Device to use
        teacher_weights: Dict of teacher weight tensors
        stage_idx: Current stage index
        is_final_stage: Whether this is the final compression stage
        substituted_keys: Optional set of module keys that were substituted
    """
    if substituted_keys is None:
        substituted_keys = set()

    if is_final_stage:
        print("\n🔚 Final stage, skipping densify, keeping low rank truncations as final before consolidation.")
        return

    densified_map = densify_target_modules(args.model, model, verbose=args.verbose)

    # Free old LowRankLinear modules to release GPU memory
    for mod in densified_map.values():
        for p in mod.parameters():
            p.data = p.data.cpu()
    del densified_map
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # Alpha blend correction toward teacher after densification
    if getattr(args, "alpha_blend_correction", False):
        # Evaluate PPL before alpha_blend
        print(f"\n📊 Evaluating PPL BEFORE alpha_blend_correction (stage {stage_idx})...")
        model.eval()
        ppl_before_dict = ppl_eval(model, tokenizer, ["wikitext2"], args.model_seq_len)
        ppl_before = ppl_before_dict["wikitext2"]
        print(f"   PPL before alpha_blend_correction: {ppl_before:.4f}")
        model.train()

        run_alpha_blend_between_stages(
            args, model, dev, teacher_weights, stage_idx=stage_idx,
            substituted_keys=substituted_keys,
        )

        # Evaluate PPL after alpha_blend
        print(f"\n📊 Evaluating PPL AFTER alpha_blend_correction (stage {stage_idx})...")
        model.eval()
        ppl_after_dict = ppl_eval(model, tokenizer, ["wikitext2"], args.model_seq_len)
        ppl_after = ppl_after_dict["wikitext2"]
        print(f"   PPL after alpha_blend_correction: {ppl_after:.4f}")
        print(f"   Delta PPL (alpha_blend): {ppl_after - ppl_before:+.4f}")
        model.train()

    # Project to delta correction
    if getattr(args, "project_to_delta", False):
        # Evaluate PPL before and after project_to_delta
        print(f"\n📊 Evaluating PPL BEFORE project_to_delta (stage {stage_idx})...")
        model.eval()
        ppl_before_dict = ppl_eval(model, tokenizer, ["wikitext2"], args.model_seq_len)
        ppl_before = ppl_before_dict["wikitext2"]
        print(f"   PPL before project_to_delta: {ppl_before:.4f}")
        model.train()

        run_project_to_delta_between_stages(
            args, model, tokenizer, dev, teacher_weights, stage_idx=stage_idx,
            substituted_keys=substituted_keys,
        )

        print(f"\n📊 Evaluating PPL AFTER project_to_delta (stage {stage_idx})...")
        model.eval()
        ppl_after_dict = ppl_eval(model, tokenizer, ["wikitext2"], args.model_seq_len)
        ppl_after = ppl_after_dict["wikitext2"]
        print(f"   PPL after project_to_delta: {ppl_after:.4f}")
        print(f"   Delta PPL (project_to_delta): {ppl_after - ppl_before:+.4f}")
        model.train()

    # Existing optional between stage correction
    if getattr(args, "pull_subspace", False):
        run_pull_subspace_between_stages(
            args, model, tokenizer, dev, teacher_weights, stage_idx=stage_idx,
            substituted_keys=substituted_keys,
        )
