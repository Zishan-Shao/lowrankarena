"""
adasvd_wrapper.py  (paper-compliant, NAACL 2024)

Changes vs adasvd_refactored:
  1. train_adasvd_ranks(): PaperHN (fixed-z, LayerNorm, meta_proj)
  2. Hard/soft mask split: hard → task forward; soft → budget + alignment losses
  3. parameter_budget() receives SOFT masks, returns scalar log (paper Eq.8)
  4. λ=16, γ=10 (paper defaults); steps=800 default
  5. Calibration: shuffled batch-level random subset up to max_calib_samples
  6. Freeze order: original params → replace_with_masked → freeze again → only HN trained
  7. Diagnostic logging: log-form budget loss, ratio_soft vs target

compress_adasvd_naive / compress_adasvd_flashsvd are unchanged from adasvd_refactored.
"""

import os
import sys
import json
import random
import torch
import torch.nn as nn
from typing import Dict, Optional
from transformers import PreTrainedModel
from torch.utils.data import DataLoader

# Add current directory so local adaptive_rank_selection is found
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


def train_adasvd_ranks(
    model: PreTrainedModel,
    calib_loader: DataLoader,
    budget: float,
    output_dir: str = "ars_out",
    steps: int = 800,
    max_calib_samples: int = 4000,
    engineering_stable: bool = False,
    seed: int = 42,
    device: str = "cuda",
) -> Dict[str, int]:
    """
    Train paper-compliant AdaSVD hypernetwork (PaperHN) to generate per-op ranks.

    Args:
        model: HuggingFace model (BertForSequenceClassification etc.)
        calib_loader: Full training DataLoader (will sample up to max_calib_samples)
        budget: Target parameter ratio (e.g. 0.5 = 50% of original Linear params)
        output_dir: Directory to save ranks.json and budget_report.json
        steps: Number of HN training steps (paper: 800)
        max_calib_samples: Max calibration samples (paper: ~4000, batch-level shuffle)
        engineering_stable: If True, use learned alpha_z gate in PaperHN (ablation)
        seed: Random seed for reproducible batch shuffle
        device: torch device string

    Returns:
        Dict mapping Linear module names to integer ranks
    """
    from adaptive_rank_selection import (
        set_seed, collect_linear_modules, replace_with_masked,
        PaperHN, collect_op_metadata,
        gumbel_sigmoid, parameter_budget, alignment_loss,
    )

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Freeze original model ─────────────────────────────────────────
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device).eval()

    # ── Step 2: Collect calibration batches (batch-level shuffle) ─────────────
    # Collect 2× target, then shuffle, then slice to max_calib_samples
    collected = []
    n_total = 0
    for b in calib_loader:
        collected.append({k: v for k, v in b.items()})
        n_total += b["input_ids"].size(0)
        if n_total >= 2 * max_calib_samples:
            break

    rng = random.Random(seed)
    rng.shuffle(collected)

    all_batches, n_used = [], 0
    for b in collected:
        all_batches.append(b)
        n_used += b["input_ids"].size(0)
        if n_used >= max_calib_samples:
            break

    actual_samples = sum(b["input_ids"].size(0) for b in all_batches)
    print(f"[ARS] Calibration: {actual_samples} samples "
          f"({len(all_batches)} batches, target={max_calib_samples}, batch-level shuffle, seed={seed})")

    # ── Step 3: Collect Linear modules and metadata ───────────────────────────
    # Exclude task head (classifier) and pooler from ARS — kept full-rank.
    # Reasons:
    #   classifier: rank_cap = num_labels (e.g. 2), ARS assigns rank=1 (minimum),
    #               crippling post-compression fine-tuning for binary tasks.
    #   pooler:     task-specific [CLS] transform; compression degrades representation
    #               quality for fine-grained tasks (e.g. CoLA linguistic acceptability).
    # compress_adasvd_naive already handles "not in ranks_dict → leave as-is".
    HEAD_EXCLUDE = ("classifier", "pooler")
    linear_list_all = collect_linear_modules(model)
    linear_list = [(n, m) for n, m in linear_list_all
                   if not any(pat in n for pat in HEAD_EXCLUDE)]
    excluded = [n for n, _ in linear_list_all if any(pat in n for pat in HEAD_EXCLUDE)]
    if excluded:
        print(f"[ARS] Excluded from ARS (task head / pooler, kept full-rank): {excluded}")
    op_names    = [n for n, _ in linear_list]
    R_caps      = [min(lin.in_features, lin.out_features) for _, lin in linear_list]
    op_metadata = collect_op_metadata(linear_list)

    # Debug: log op-type distribution
    from collections import Counter
    type_names = ["q", "k", "v", "attn_out", "intermediate", "ffn_out", "other"]
    type_counts = Counter()
    for meta_row in op_metadata:
        hot = meta_row[2:]
        idx = hot.index(1.0) if 1.0 in hot else 6
        type_counts[type_names[idx]] += 1
    print(f"[ARS] op_type distribution: {dict(type_counts)}")
    print(f"[ARS] Expected for BERT-base (12 layers): q:12, k:12, v:12, attn_out:12, intermediate:12, ffn_out:12")

    # ── Step 4: Replace Linear → MaskedSVDLinear ─────────────────────────────
    print(f"[ARS] Replacing {len(op_names)} Linear layers with MaskedSVDLinear ...")
    model, masked_ops = replace_with_masked(model, device=device,
                                            rank_cap_per_op=R_caps,
                                            original_names=op_names)
    op_list = [op for _, op in masked_ops]

    # ── Step 5: Freeze again (covers new MaskedSVDLinear params too) ──────────
    for p in model.parameters():
        p.requires_grad_(False)

    # ── Step 6: Build PaperHN and optimizer ───────────────────────────────────
    HN = PaperHN(op_sizes=R_caps, op_metadata=op_metadata,
                 engineering_stable=engineering_stable,
                 budget=budget).to(device)
    optimizer = torch.optim.Adam(HN.parameters(), lr=1e-3)

    # Pre-compute T_original once (encoder Linear layers only, NOT embeddings/LayerNorm/classifier/pooler)
    T_original = sum(op.in_features * op.out_features for op in op_list)

    # ratio_max: ratio_soft at FULL RANK (all singular values kept).
    # Formula: sum((in+out)*Rcap) / sum(in*out). For square layers (M=N), this = 2.
    # For BERT-base mix (square attn + rect FFN): ratio_max ≈ 1.50.
    # Any ratio_soft > ratio_max indicates a formula inconsistency (e.g., wrong
    # denominator or duplicate op counting), NOT just near-full-rank operation.
    # op.rank_cap = effective rank cap stored in MaskedSVDLinear (= min(rank_cap, Rfull)).
    # Using op.rank_cap (not a freshly-computed min(in,out)) ensures we stay consistent
    # with what was actually passed to replace_with_masked.
    T_max_fullrank = sum((op.in_features + op.out_features) * op.rank_cap for op in op_list)
    ratio_max = T_max_fullrank / (T_original + 1e-12)
    print(f"[ARS] ratio_max (full-rank SVD/dense) = {ratio_max:.3f}  "
          f"(target budget = {budget:.3f})")

    # ── Step 7: Training loop ─────────────────────────────────────────────────
    print(f"[ARS] Training PaperHN for {steps} steps (budget={budget}, λ=100, γ=10) ...")
    batch_idx = 0
    for step in range(steps):
        batch = all_batches[batch_idx % len(all_batches)]
        batch_idx += 1

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        # Emit logits from PaperHN
        logits_list = HN()

        # Hard Gumbel-Sigmoid: discrete binary mask for task forward (paper Eq.6)
        # [Hard rule B]: ONLY hard masks go to _current_mask
        masks_hard = [gumbel_sigmoid(l, tau=1.0, hard=True) for l in logits_list]

        # Soft sigmoid: for budget + alignment losses (clean, low-variance gradients)
        masks_soft = [torch.sigmoid(l) for l in logits_list]

        # Set HARD masks for model forward pass
        for op, m in zip(op_list, masks_hard):
            # [Hard rule C]: dtype/device safety
            op._current_mask = m.to(op.U.dtype).to(op.U.device)

        # Task loss (uses hard masks via _current_mask)
        outputs   = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        task_loss = outputs.loss

        # Budget loss: SOFT masks → scalar log (paper Eq.8)
        budget_loss = parameter_budget(op_list, masks_soft, budget)

        # Alignment loss: k from HARD mask, gradient via SOFT mask (paper Eq.7 spirit).
        # k_ref=m_hard.detach() breaks the self-lock: budget pushes logits down →
        # hard k decreases → alignment must also pull soft mask lower → feedback unblocked.
        align_loss = sum(
            alignment_loss(m_soft, op.s, k_ref=m_hard.detach())
            for op, m_soft, m_hard in zip(op_list, masks_soft, masks_hard)
        )

        # λ=100 (budget强主导), γ=10 (alignment辅助)
        loss = task_loss + 100.0 * budget_loss + 10.0 * align_loss

        loss.backward()
        optimizer.step()

        # Clear masks after step
        for op in op_list:
            op._current_mask = None

        if step % 50 == 0:
            with torch.no_grad():
                # ratio_soft uses SAME T_original (pre-computed, Linear only)
                ratio_soft = sum(
                    (op.in_features + op.out_features) * torch.sigmoid(l).sum().item()
                    for op, l in zip(op_list, logits_list)
                ) / T_original
            # Sanity check: ratio_soft must never exceed ratio_max (full-rank SVD/dense).
            # If it does, T or T_original formula is inconsistent (wrong units or dup ops).
            assert ratio_soft <= ratio_max + 1e-2, (
                f"ratio_soft={ratio_soft:.4f} exceeds ratio_max={ratio_max:.4f} — "
                f"T or T_original definition is inconsistent (wrong units or duplicate ops)."
            )
            print(f"  step={step:4d} | task={task_loss.item():.4f} "
                  f"| budget={budget_loss.item():.4f}(log) | align={align_loss.item():.4f} "
                  f"| ratio_soft={ratio_soft:.3f} target={budget:.3f} max={ratio_max:.3f}")

    # ── Step 8: Extract integer ranks from soft masks ─────────────────────────
    print("[ARS] Finalizing ranks from soft masks ...")
    with torch.no_grad():
        logits_list = HN()
        masks_soft  = [torch.sigmoid(l) for l in logits_list]
        ks = [int(torch.clamp(torch.round(m.sum()), 1, m.numel()).item())
              for m in masks_soft]

    ranks_dict = {name: int(k) for name, k in zip(op_names, ks)}

    # Achieved ratio (same T_original definition)
    total_svd_params = sum((op.in_features + op.out_features) * k
                           for op, k in zip(op_list, ks))
    achieved_ratio = total_svd_params / T_original

    # ── Step 9: Save outputs ──────────────────────────────────────────────────
    ranks_path = os.path.join(output_dir, "ranks.json")
    with open(ranks_path, "w") as f:
        json.dump(ranks_dict, f, indent=2)
    print(f"[ARS] Saved ranks to: {ranks_path}")

    budget_report = {
        "target_budget": budget,
        "achieved_ratio": achieved_ratio,
        # Keys match what _calculate_param_ratio() in run_encoder_benchmark.py reads:
        "total_params": int(total_svd_params),           # "kept" SVD params
        "original_model_params": int(T_original),         # original Linear in×out params
        # Extra diagnostics (paper-origin specific):
        "original_linear_params": int(T_original),        # alias for clarity
        "total_svd_params": int(total_svd_params),        # alias for clarity
        "num_operations": len(op_list),
        "steps": steps,
        "max_calib_samples": max_calib_samples,
        "actual_calib_samples": actual_samples,
        "engineering_stable": engineering_stable,
        "hn_type": "PaperHN",
    }
    report_path = os.path.join(output_dir, "budget_report.json")
    with open(report_path, "w") as f:
        json.dump(budget_report, f, indent=2)
    print(f"[ARS] Saved budget report to: {report_path}")
    print(f"[ARS] Target: {budget:.3f} | Achieved: {achieved_ratio:.3f}")

    return ranks_dict


# ── Compression functions (unchanged from adasvd_refactored) ─────────────────

class _LowRankLinear(nn.Module):
    """
    Drop-in nn.Linear replacement: W ≈ A @ Bt where A=[out,r], Bt=[r,in].
    Forward: x @ Bt.T @ A.T + bias  (= x @ W.T + bias at full rank).

    Semantics match ARS exactly: ARS trains per-Linear full-matrix masked SVD;
    we reconstruct the same full-matrix low-rank factorisation here.
    No Triton kernels, no per-head reshaping — pure PyTorch matmul.
    """
    def __init__(self, A: torch.Tensor, Bt: torch.Tensor,
                 bias: "torch.Tensor | None" = None):
        super().__init__()
        self.A    = nn.Parameter(A,  requires_grad=False)   # [out, r]
        self.Bt   = nn.Parameter(Bt, requires_grad=False)   # [r,  in]
        self.bias = nn.Parameter(bias.detach().clone(),
                                 requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mid = x @ self.Bt.t()                             # [..., r]
        out = mid @ self.A.t()                            # [..., out]
        if self.bias is not None:
            out = out + self.bias
        return out


def compress_adasvd_naive(
    model: PreTrainedModel,
    ranks_path: str,
    device: str = "cuda",
) -> PreTrainedModel:
    """
    Compress model with AdaSVD naive backend: per-Linear full-matrix SVD.

    Each nn.Linear in ranks_dict is replaced with _LowRankLinear(A, Bt) where
    W ≈ A @ Bt, A = U_r * S_r  [out, r],  Bt = Vh_r  [r, in].

    This matches ARS semantics exactly (ARS trained per-Linear masked SVD with
    full-matrix rank budget).  The HF model structure (attention, FFN, pooler,
    classifier) is preserved unchanged — no Triton kernels, no per-head
    reshaping that would misinterpret the ARS per-op ranks.
    """
    model = model.to(device).eval()
    with open(ranks_path, "r") as f:
        ranks_dict = json.load(f)
    print(f"[adasvd_origin] Loaded {len(ranks_dict)} per-op ranks from {ranks_path}")

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        rank = ranks_dict.get(name)
        if rank is None:
            continue  # classifier / pooler excluded from ARS → keep full-rank

        W    = module.weight.data.float()           # [out, in]
        rank = max(1, min(rank, min(W.shape)))      # clamp to valid range

        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        A  = (U[:, :rank] * S[:rank]).to(module.weight.dtype)   # [out, r]
        Bt = Vh[:rank, :].to(module.weight.dtype)               # [r,  in]

        bias = module.bias  # None or Parameter
        lrl  = _LowRankLinear(A, Bt, bias).to(device)

        # Navigate to parent and swap the module
        *parts, last = name.split(".")
        parent = model
        for part in parts:
            parent = getattr(parent, part)
        setattr(parent, last, lrl)
        replaced += 1

    print(f"[adasvd_origin] Replaced {replaced} Linear ops with LowRankLinear (naive, full-matrix SVD)")
    torch.cuda.empty_cache()
    return model


def compress_adasvd_flashsvd(
    model: PreTrainedModel,
    ranks_path: str,
    ffn_kernel: str = "v1",
    device: str = "cuda",
) -> PreTrainedModel:
    """Compress model with AdaSVD using FlashSVD backend (Triton kernels)."""
    import numpy as np

    _refactored = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "adasvd_refactored"
    )
    if _refactored not in sys.path:
        sys.path.insert(0, _refactored)

    from profile_flashsvd import (
        attach_fullnames, build_plain_svd_helpers, FlashSVDBlock, LayerShim
    )

    model = model.to(device).eval()
    with open(ranks_path, "r") as f:
        ranks_dict = json.load(f)
    print(f"[adasvd_origin] Loaded {len(ranks_dict)} per-op ranks from {ranks_path}")

    # Median-rank strategy for low budgets (FlashSVD requires uniform ranks)
    ranks_dir = os.path.dirname(ranks_path)
    budget_report_path = os.path.join(ranks_dir, "budget_report.json")
    if os.path.exists(budget_report_path):
        with open(budget_report_path, "r") as f:
            budget_report = json.load(f)
        target_budget = budget_report.get("target_budget", 1.0)
        if target_budget < 0.3:
            ranks_list  = [r for r in ranks_dict.values() if r > 0]
            median_rank = int(np.median(ranks_list))
            ranks_dict  = {k: median_rank for k in ranks_dict.keys()}
            print(f"[adasvd_origin] FlashSVD low-budget: using median rank={median_rank} uniformly")
        else:
            print(f"[adasvd_origin] FlashSVD per-op ranks (budget={target_budget:.2f})")

    attach_fullnames(model)
    svd_per_head, svd_low_rank = build_plain_svd_helpers(model)

    if hasattr(model, 'bert'):
        encoder_layers = model.bert.encoder.layer
    elif hasattr(model, 'roberta'):
        encoder_layers = model.roberta.encoder.layer
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    print(f"[adasvd_origin] Replacing {len(encoder_layers)} layers with FlashSVDBlock (ffn={ffn_kernel}) ...")
    for i, layer in enumerate(encoder_layers):
        block   = FlashSVDBlock(layer, ranks_dict, svd_per_head, svd_low_rank, ffn_kernel=ffn_kernel)
        shimmed = LayerShim(block)
        encoder_layers[i] = shimmed.to(device).eval()

    del svd_per_head, svd_low_rank
    torch.cuda.empty_cache()
    return model


def get_param_ratio_from_budget_report(output_dir: str = "ars_out") -> float:
    report_path = os.path.join(output_dir, "budget_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = json.load(f)
        return report.get("achieved_ratio", 0.5)
    return 0.5
