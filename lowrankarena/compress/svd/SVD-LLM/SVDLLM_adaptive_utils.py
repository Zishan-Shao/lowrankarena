#coding:utf8
"""
Shared adaptive rank-allocation helpers for the local SVD-LLM v2 path.

Implementation boundary:
- This file only provides local helper utilities for adaptive/module-wise
  rank allocation and whitening-aware replacement.
- It is not claimed to be a line-for-line equivalent of any upstream official
  implementation.
"""

from __future__ import annotations

import heapq
import math
import os
import sys
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

from SVDLLM import (  # noqa: E402
    _maybe_release_guard,
    _maybe_reserve_guard,
    _rank_from_keep_ratio,
)
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP  # noqa: E402
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP  # noqa: E402
from component.svd_opt import SVDOPTDecoderLayer  # noqa: E402
from utils.model_utils import find_layers  # noqa: E402


IMPLEMENTATION_STATUS = "local_adaptive_helper_layer"
IMPLEMENTATION_BOUNDARY = (
    "This module contains local helper utilities for the adaptive/module-wise "
    "SVD-LLM v2 path. It is not claimed to be a line-for-line equivalent of "
    "an upstream official implementation."
)

_ATTN_NAME_HINTS = ("q_proj", "k_proj", "v_proj", "o_proj", "out_proj")
_MLP_NAME_HINTS = ("gate_proj", "down_proj", "up_proj", "fc1", "fc2")


def _get_layers(model_name, model):
    if "opt" in model_name:
        return model.model.decoder.layers
    if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
        return model.model.layers
    raise ValueError(f"Unsupported model name: {model_name}")


def _module_group(name: str) -> str:
    module_name = str(name)
    if any(hint in module_name for hint in _ATTN_NAME_HINTS):
        return "attn"
    if any(hint in module_name for hint in _MLP_NAME_HINTS):
        return "mlp"
    return "other"


def _module_type_group(name: str) -> str:
    module_name = str(name)
    if "q_proj" in module_name:
        return "q_proj"
    if "k_proj" in module_name:
        return "k_proj"
    if "v_proj" in module_name:
        return "v_proj"
    if "o_proj" in module_name or "out_proj" in module_name:
        return "o_proj"
    if "gate_proj" in module_name:
        return "gate_proj"
    if "up_proj" in module_name:
        return "up_proj"
    if "down_proj" in module_name:
        return "down_proj"
    if "fc1" in module_name:
        return "fc1"
    if "fc2" in module_name:
        return "fc2"
    return module_name


def _iter_target_modules(layer):
    for name, module in find_layers(layer).items():
        if isinstance(module, nn.Linear):
            yield name, module


def _sync_linear_meta(lin_mod: nn.Linear):
    if not isinstance(lin_mod, nn.Linear):
        return
    lin_mod.in_features = int(lin_mod.weight.shape[1])
    lin_mod.out_features = int(lin_mod.weight.shape[0])
    if lin_mod.bias is not None and lin_mod.bias.numel() != lin_mod.out_features:
        new_bias = lin_mod.bias.new_zeros(lin_mod.out_features)
        copy_size = min(int(lin_mod.bias.numel()), int(lin_mod.out_features))
        new_bias[:copy_size] = lin_mod.bias.data[:copy_size]
        lin_mod.bias = nn.Parameter(new_bias, requires_grad=lin_mod.bias.requires_grad)


def _resolve_module_keep_ratio(
    layer_idx: int,
    module_name: str,
    ratio: float,
    *,
    attn_ratio: float | None = None,
    mlp_ratio: float | None = None,
    module_keep_ratios: Optional[Dict[Tuple[int, str], float]] = None,
) -> float:
    key = (int(layer_idx), str(module_name))
    if module_keep_ratios is not None and key in module_keep_ratios:
        return float(module_keep_ratios[key])
    group = _module_group(module_name)
    if group == "attn" and attn_ratio is not None:
        return float(attn_ratio)
    if group == "mlp" and mlp_ratio is not None:
        return float(mlp_ratio)
    return float(ratio)


def _module_param_cost(module: nn.Linear) -> int:
    return int(module.in_features) + int(module.out_features)


def _module_dense_params(module: nn.Linear) -> int:
    return int(module.in_features) * int(module.out_features)


def _module_rank_budget(module: nn.Linear, keep_ratio: float, force_param_count_rank: bool = True) -> int:
    keep_ratio = min(1.0, max(0.0, float(keep_ratio)))
    if force_param_count_rank:
        return _rank_from_keep_ratio(module.out_features, module.in_features, keep_ratio)
    return max(1, int(min(module.out_features, module.in_features) * keep_ratio))


def _module_keep_ratio_from_rank(module: nn.Linear, rank: int) -> float:
    rank = max(1, int(rank))
    dense = max(1, _module_dense_params(module))
    return min(1.0, max(0.0, rank * _module_param_cost(module) / dense))


def _module_max_param_rank(module: nn.Linear) -> int:
    return max(1, _rank_from_keep_ratio(module.out_features, module.in_features, 1.0))


def _svdvals_with_fallback(weight: torch.Tensor, dev) -> torch.Tensor:
    try:
        return torch.linalg.svdvals(weight.to(dev, dtype=torch.float32))
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
        return torch.linalg.svdvals(weight.to("cpu", dtype=torch.float32))


def _profile_projected_weight(weight: torch.Tensor, profile_entry, dev) -> torch.Tensor:
    weight = weight.to(dev, dtype=torch.float32)
    if isinstance(profile_entry, dict) and "u" in profile_entry and "root" in profile_entry:
        basis = profile_entry["u"].to(dev, dtype=torch.float32)
        root = profile_entry["root"].to(dev, dtype=torch.float32)
        return weight @ (basis * root.unsqueeze(0))
    scaling = profile_entry.to(dev, dtype=torch.float32)
    return weight @ scaling


def _profile_inverse_right(vt: torch.Tensor, profile_entry, dev) -> torch.Tensor:
    vt = vt.to(dev, dtype=torch.float32)
    if isinstance(profile_entry, dict) and "u" in profile_entry and "inv_root" in profile_entry:
        basis = profile_entry["u"].to(dev, dtype=torch.float32)
        inv_root = profile_entry["inv_root"].to(dev, dtype=torch.float32)
        return (vt * inv_root.unsqueeze(0)) @ basis.t()
    scaling = profile_entry.to(dev, dtype=torch.float32)
    return vt @ torch.linalg.pinv(scaling)


def _tail_fro_loss(singular_values: torch.Tensor, rank: int) -> float:
    rank = max(0, int(rank))
    if rank >= int(singular_values.numel()):
        return 0.0
    tail = singular_values[rank:]
    return float(torch.linalg.vector_norm(tail, ord=2).item())


def _paper_loss_score(loss: float) -> float:
    loss = max(float(loss), 1e-12)
    if loss > 1.0 + 1e-12:
        denom = math.log(loss)
    else:
        # The paper writes 1/log(Lmin), but its reported normalized losses can be < 1.
        # Use log1p as a stability fallback in that regime while keeping the same inverse-log intent.
        denom = math.log1p(loss)
    return 1.0 / max(denom, 1e-12)


def _improve_rank_budget_to_target(module_entries, target_budget: int):
    current_budget = sum(int(entry["rank"]) * int(entry["cost"]) for entry in module_entries)
    if current_budget == target_budget:
        return

    if current_budget < target_budget:
        heap = []
        for idx, entry in enumerate(module_entries):
            if int(entry["rank"]) < int(entry["max_rank"]):
                gain = float(entry["singular_values"][int(entry["rank"])].item() ** 2) / float(entry["cost"])
                heapq.heappush(heap, (-gain, idx, int(entry["rank"])))
        while heap:
            _, idx, expected_rank = heapq.heappop(heap)
            entry = module_entries[idx]
            if int(entry["rank"]) != int(expected_rank):
                continue
            next_budget = current_budget + int(entry["cost"])
            if abs(next_budget - target_budget) >= abs(current_budget - target_budget):
                continue
            entry["rank"] += 1
            current_budget = next_budget
            if int(entry["rank"]) < int(entry["max_rank"]):
                gain = float(entry["singular_values"][int(entry["rank"])].item() ** 2) / float(entry["cost"])
                heapq.heappush(heap, (-gain, idx, int(entry["rank"])))
    else:
        heap = []
        for idx, entry in enumerate(module_entries):
            if int(entry["rank"]) > 1:
                lost = float(entry["singular_values"][int(entry["rank"]) - 1].item() ** 2) / float(entry["cost"])
                heapq.heappush(heap, (lost, idx, int(entry["rank"])))
        while heap:
            _, idx, expected_rank = heapq.heappop(heap)
            entry = module_entries[idx]
            if int(entry["rank"]) != int(expected_rank):
                continue
            next_budget = current_budget - int(entry["cost"])
            if abs(next_budget - target_budget) >= abs(current_budget - target_budget):
                continue
            entry["rank"] -= 1
            current_budget = next_budget
            if int(entry["rank"]) > 1:
                lost = float(entry["singular_values"][int(entry["rank"]) - 1].item() ** 2) / float(entry["cost"])
                heapq.heappush(heap, (lost, idx, int(entry["rank"])))


@torch.no_grad()
def allocate_weight_type_keep_ratios(
    model_name,
    model,
    profiling_mat,
    target_reduction_ratio: float,
    dev,
    strict_formula: bool = True,
    implementation_label: str = "adaptive",
):
    layers = _get_layers(model_name, model)
    target_keep_ratio = 1.0 - float(target_reduction_ratio)
    if not (0.0 < target_keep_ratio <= 1.0):
        raise ValueError(f"Expected keep ratio in (0, 1], got {target_keep_ratio}")

    module_entries = []
    grouped_entries = defaultdict(list)
    total_dense_params = 0
    target_budget = 0

    for layer_idx, layer in enumerate(layers):
        for module_name, module in _iter_target_modules(layer):
            dense_params = _module_dense_params(module)
            total_dense_params += dense_params
            profile_entry = profiling_mat[layer_idx][module_name]
            weight_scale = _profile_projected_weight(module.weight.data, profile_entry, dev=dev)
            singular_values = _svdvals_with_fallback(weight_scale, dev=dev).cpu()
            base_rank = min(
                int(singular_values.numel()),
                _module_max_param_rank(module),
                _module_rank_budget(module, target_keep_ratio),
            )
            lmin = _tail_fro_loss(singular_values, base_rank)
            target_budget += int(base_rank) * _module_param_cost(module)

            entry = {
                "key": (layer_idx, module_name),
                "module": module,
                "type_group": _module_type_group(module_name),
                "singular_values": singular_values,
                "cost": _module_param_cost(module),
                "rank": base_rank,
                "max_rank": min(int(singular_values.numel()), _module_max_param_rank(module)),
                "lmin": lmin,
            }
            module_entries.append(entry)
            grouped_entries[entry["type_group"]].append(entry)

            del weight_scale
            if str(dev).startswith("cuda"):
                torch.cuda.empty_cache()

    for _, group_entries in grouped_entries.items():
        scores = []
        for entry in group_entries:
            score = _paper_loss_score(entry["lmin"]) if strict_formula else max(float(entry["lmin"]), 1e-12)
            scores.append(score)
        score_sum = sum(scores) if sum(scores) > 0 else float(len(scores))
        group_size = len(group_entries)
        for entry, score in zip(group_entries, scores):
            reduce_ratio = float(group_size) * float(target_reduction_ratio) * float(score) / float(score_sum)
            keep_ratio = min(1.0, max(0.0, 1.0 - reduce_ratio))
            entry["rank"] = min(
                int(entry["max_rank"]),
                _module_rank_budget(entry["module"], keep_ratio),
            )

    _improve_rank_budget_to_target(module_entries, target_budget)

    module_keep_ratios = {}
    module_reduce_ratios = {}
    module_lmin = {}
    for entry in module_entries:
        keep_ratio = _module_keep_ratio_from_rank(entry["module"], entry["rank"])
        key = entry["key"]
        module_keep_ratios[key] = keep_ratio
        module_reduce_ratios[key] = 1.0 - keep_ratio
        module_lmin[key] = float(entry["lmin"])

    achieved_budget = sum(
        _module_rank_budget(e["module"], module_keep_ratios[e["key"]]) * _module_param_cost(e["module"])
        for e in module_entries
    )
    achieved_keep_ratio = achieved_budget / max(1, total_dense_params)
    print(
        f"[{implementation_label}] adaptive allocation target_keep_ratio={target_keep_ratio:.4f} "
        f"achieved_keep_ratio={achieved_keep_ratio:.4f} modules={len(module_entries)}"
    )
    return module_keep_ratios, module_reduce_ratios, module_lmin


@torch.no_grad()
def apply_module_keep_ratios(
    model_name,
    model,
    profiling_mat,
    ratio,
    dev,
    *,
    attn_ratio: float = None,
    mlp_ratio: float = None,
    svd_method: str = "full",
    svd_niter: int = 2,
    svd_oversample: int = 5,
    module_keep_ratios: Optional[Dict[Tuple[int, str], float]] = None,
    force_param_count_rank: bool = True,
    implementation_label: str = "adaptive",
    gpu_guard=None,
):
    model.eval()
    layers = _get_layers(model_name, model)
    svd_method = (svd_method or "full").lower()
    print(f"[{implementation_label}] start module-wise whitening + SVD")

    for layer_idx in tqdm(range(len(layers))):
        _maybe_release_guard(gpu_guard, f"{implementation_label} layer {layer_idx}")
        layer = layers[layer_idx]
        subset = dict(_iter_target_modules(layer))

        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=1.0)
            svd_mlp = SVD_LlamaMLP(
                hidden_size=layer.hidden_size,
                intermediate_size=model.config.intermediate_size,
                hidden_act=model.config.hidden_act,
                ratio=1.0,
            )
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=1.0)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=1.0)
        elif "opt" in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=1.0)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        for module_name, module in subset.items():
            keep_ratio = _resolve_module_keep_ratio(
                layer_idx,
                module_name,
                ratio,
                attn_ratio=attn_ratio,
                mlp_ratio=mlp_ratio,
                module_keep_ratios=module_keep_ratios,
            )
            rank = _module_rank_budget(module, keep_ratio, force_param_count_rank=force_param_count_rank)

            rank = min(rank, _module_max_param_rank(module))
            dtype = module.weight.dtype
            profile_entry = profiling_mat[layer_idx][module_name]
            weight_scale = _profile_projected_weight(module.weight.data, profile_entry, dev=dev)

            if svd_method == "randomized":
                q = min(rank + max(0, int(svd_oversample)), min(weight_scale.shape))
                U, S, V = torch.svd_lowrank(weight_scale.to(dtype=torch.float32), q=q, niter=max(0, int(svd_niter)))
                VT = V[:, :rank].T
                U = U[:, :rank].to(dtype=torch.float32)
                S = S[:rank].to(dtype=torch.float32)
                VT = VT.to(dtype=torch.float32)
            else:
                U, S, VT = torch.linalg.svd(weight_scale, full_matrices=False)
                U = U[:, :rank]
                S = S[:rank]
                VT = VT[:rank, :]

            trunc_v = _profile_inverse_right(VT, profile_entry, dev=dev)
            sqrt_sigma = torch.sqrt(torch.diag(S))
            svd_u = torch.matmul(U, sqrt_sigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrt_sigma, trunc_v).cpu().to(dtype)

            if "opt" in model_name:
                if "q_proj" in module_name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in module_name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in module_name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                elif "out_proj" in module_name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                elif "fc1" in module_name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                elif "fc2" in module_name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[layer_idx] = svd_decoder
            else:
                if "q_proj" in module_name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in module_name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in module_name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in module_name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn = svd_attn
                elif "gate_proj" in module_name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in module_name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in module_name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp

            for candidate in (
                getattr(svd_attn, "q_u_proj", None) if "opt" not in model_name else None,
                getattr(svd_attn, "q_v_proj", None) if "opt" not in model_name else None,
                getattr(svd_attn, "k_u_proj", None) if "opt" not in model_name else None,
                getattr(svd_attn, "k_v_proj", None) if "opt" not in model_name else None,
                getattr(svd_attn, "v_u_proj", None) if "opt" not in model_name else None,
                getattr(svd_attn, "v_v_proj", None) if "opt" not in model_name else None,
                getattr(svd_attn, "o_u_proj", None) if "opt" not in model_name else None,
                getattr(svd_attn, "o_v_proj", None) if "opt" not in model_name else None,
                getattr(svd_mlp, "gate_u_proj", None) if "opt" not in model_name else None,
                getattr(svd_mlp, "gate_v_proj", None) if "opt" not in model_name else None,
                getattr(svd_mlp, "down_u_proj", None) if "opt" not in model_name else None,
                getattr(svd_mlp, "down_v_proj", None) if "opt" not in model_name else None,
                getattr(svd_mlp, "up_u_proj", None) if "opt" not in model_name else None,
                getattr(svd_mlp, "up_v_proj", None) if "opt" not in model_name else None,
            ):
                if candidate is not None:
                    _sync_linear_meta(candidate)

            del weight_scale, U, S, VT, trunc_v, sqrt_sigma
            if str(dev).startswith("cuda"):
                torch.cuda.empty_cache()

        _maybe_reserve_guard(gpu_guard, f"{implementation_label} layer {layer_idx} finished")
