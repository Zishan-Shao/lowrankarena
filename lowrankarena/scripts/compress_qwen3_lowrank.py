#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from huggingface_hub import save_torch_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELING_ROOT = REPO_ROOT / "src" / "modeling"
SVDLLM_ROOT = REPO_ROOT / "compress" / "svd" / "SVD-LLM"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SVDLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(SVDLLM_ROOT))

from src.modeling.qwen.configuration_lowrank_qwen3 import LowRankQwen3Config  # noqa: E402


TARGET_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

SHARED_BASIS_MODULES = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
}

PRIVATE_BASIS_MODULES = {
    "self_attn.o_proj",
    "mlp.down_proj",
}


def _module_key(layer_idx: int, module_name: str) -> str:
    return f"model.layers.{layer_idx}.{module_name}"


def _target_layers(model) -> nn.ModuleList:
    model_type = getattr(model.config, "model_type", "")
    if model_type != "qwen3":
        raise ValueError(f"Expected a qwen3 base model, got model_type={model_type!r}")
    return model.model.layers


def _named_target_linears(model) -> Iterable[tuple[int, str, nn.Linear]]:
    for layer_idx, layer in enumerate(_target_layers(model)):
        for module_name in TARGET_MODULES:
            module = layer.get_submodule(module_name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"{_module_key(layer_idx, module_name)} is not nn.Linear")
            yield layer_idx, module_name, module


def _rank_from_keep_ratio(in_features: int, out_features: int, keep_ratio: float) -> int:
    dense_params = int(in_features) * int(out_features)
    factor_params = int(in_features) + int(out_features)
    rank = int(dense_params * float(keep_ratio) / max(1, factor_params))
    return max(1, min(rank, int(in_features), int(out_features)))


def _basis_rank_from_keep_ratio(in_features: int, out_features: int, group_size: int, keep_ratio: float) -> int:
    dense_params = int(in_features) * int(out_features) * int(group_size)
    factor_params = int(in_features) + int(out_features) * int(group_size)
    rank = int(dense_params * float(keep_ratio) / max(1, factor_params))
    return max(1, min(rank, int(in_features), int(out_features) * int(group_size)))


def _copy_model_code(output_dir: Path) -> None:
    for filename in ("../common.py", "configuration_lowrank_qwen3.py", "modeling_lowrank_qwen3.py"):
        source = (MODELING_ROOT / "qwen" / filename).resolve()
        shutil.copy2(source, output_dir / Path(filename).name)


def _jsonify(value):
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, dict):
        return {key: _jsonify(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _build_lowrank_config(model, low_rank_modules: dict[str, dict[str, int]], method: str, extra_metadata: dict):
    config = LowRankQwen3Config.from_dict(_jsonify(model.config.to_dict()))
    config.low_rank_modules = low_rank_modules
    config.low_rank_method = method
    config.low_rank_schema = "ABLinear"
    config.low_rank_format_version = 1
    config.auto_map = {
        "AutoConfig": "configuration_lowrank_qwen3.LowRankQwen3Config",
        "AutoModel": "modeling_lowrank_qwen3.LowRankQwen3Model",
        "AutoModelForCausalLM": "modeling_lowrank_qwen3.LowRankQwen3ForCausalLM",
    }
    config.architectures = ["LowRankQwen3ForCausalLM"]
    for key, value in extra_metadata.items():
        setattr(config, key, value)
    return config


def _factorize_weight(
    weight: torch.Tensor,
    rank: int,
    device: torch.device,
    *,
    profile_entry=None,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if profile_entry is None:
        projected = weight.to(device=device, dtype=torch.float32)
        inverse_project = None
    else:
        from SVDLLM_adaptive_utils import _profile_inverse_right, _profile_projected_weight

        projected = _profile_projected_weight(weight, profile_entry, dev=device)
        inverse_project = _profile_inverse_right

    u, s, vh = torch.linalg.svd(projected, full_matrices=False)
    rank = min(int(rank), int(s.numel()))
    u = u[:, :rank]
    s = s[:rank]
    vh = vh[:rank, :]

    if inverse_project is not None:
        vh = inverse_project(vh, profile_entry, dev=device)

    sqrt_s = torch.sqrt(s)
    a_weight = (u * sqrt_s.unsqueeze(0)).to(dtype=out_dtype).cpu().contiguous()
    b_weight = (sqrt_s.unsqueeze(1) * vh).to(dtype=out_dtype).cpu().contiguous()

    del projected, u, s, vh, sqrt_s
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return a_weight, b_weight


def _replace_state_dict_entry(
    state_dict: dict[str, torch.Tensor],
    module_key: str,
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
) -> None:
    state_dict.pop(f"{module_key}.weight", None)
    bias = state_dict.pop(f"{module_key}.bias", None)
    state_dict[f"{module_key}.ALinear.weight"] = a_weight
    state_dict[f"{module_key}.BLinear.weight"] = b_weight
    if bias is not None:
        state_dict[f"{module_key}.ALinear.bias"] = bias.detach().cpu().contiguous()


def _svdllm_v2_keep_ratios(model, tokenizer, args, device: torch.device) -> dict[tuple[int, str], float]:
    from SVDLLM_v2_hetero import allocate_svdllm_v2_adaptive_keep_ratios, profile_svdllm_low_resource
    from utils.data_utils import get_calib_train_data

    model.seqlen = int(args.sequence_length)
    calib = get_calib_train_data(
        args.dataset,
        tokenizer,
        int(args.calibration_samples),
        seqlen=int(args.sequence_length),
        seed=int(args.seed),
        batch_size=1,
    )
    profiling_mat = profile_svdllm_low_resource(
        args.model_id.lower(),
        model,
        calib,
        str(device),
        raw_xtx=True,
        low_resource_factor_device="gpu",
    )
    module_keep_ratios, _, _ = allocate_svdllm_v2_adaptive_keep_ratios(
        model_name=args.model_id.lower(),
        model=model,
        profiling_mat=profiling_mat,
        target_reduction_ratio=1.0 - float(args.keep_ratio),
        dev=str(device),
        strict_paper_formula=True,
    )
    return module_keep_ratios, profiling_mat


def _compress_svdllm_v2(model, tokenizer, state_dict, args, device: torch.device):
    module_keep_ratios, profiling_mat = _svdllm_v2_keep_ratios(model, tokenizer, args, device)
    low_rank_modules: dict[str, dict[str, int]] = {}

    for layer_idx, module_name, module in _named_target_linears(model):
        keep_ratio = float(module_keep_ratios[(layer_idx, module_name)])
        rank = _rank_from_keep_ratio(module.in_features, module.out_features, keep_ratio)
        module_key = _module_key(layer_idx, module_name)
        print(f"[svdllm_v2] {module_key} keep={keep_ratio:.6f} rank={rank}", flush=True)
        a_weight, b_weight = _factorize_weight(
            module.weight.detach(),
            rank,
            device,
            profile_entry=profiling_mat[layer_idx][module_name],
            out_dtype=module.weight.dtype,
        )
        _replace_state_dict_entry(state_dict, module_key, a_weight, b_weight)
        low_rank_modules[module_key] = {"rank": int(rank)}

    metadata = {
        "svdllm_v2_implementation": "local_paper_derived_adaptive_rank_allocation",
        "svdllm_v2_calibration_dataset": args.dataset,
        "svdllm_v2_calibration_samples": int(args.calibration_samples),
        "svdllm_v2_sequence_length": int(args.sequence_length),
    }
    return low_rank_modules, metadata


def _basis_groups(num_layers: int, module_name: str, group_size: int) -> list[list[int]]:
    if module_name in PRIVATE_BASIS_MODULES:
        return [[idx] for idx in range(num_layers)]
    if module_name not in SHARED_BASIS_MODULES:
        raise ValueError(f"Unexpected Basis Sharing module: {module_name}")
    return [
        list(range(start, min(start + int(group_size), num_layers)))
        for start in range(0, num_layers, int(group_size))
    ]


def _factorize_basis_group(
    modules: list[nn.Linear],
    rank: int,
    device: torch.device,
    out_dtype: torch.dtype,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    concatenated = torch.cat([module.weight.detach().t().to(device=device, dtype=torch.float32) for module in modules], dim=1)
    u, s, vh = torch.linalg.svd(concatenated, full_matrices=False)
    rank = min(int(rank), int(s.numel()))
    u = u[:, :rank]
    s = s[:rank]
    vh = vh[:rank, :]
    sqrt_s = torch.sqrt(s)
    b_weight = (sqrt_s.unsqueeze(1) * u.t()).to(dtype=out_dtype).cpu().contiguous()

    factors = []
    offset = 0
    for module in modules:
        width = int(module.out_features)
        vh_block = vh[:, offset : offset + width]
        a_weight = (vh_block.t() * sqrt_s.unsqueeze(0)).to(dtype=out_dtype).cpu().contiguous()
        factors.append((a_weight, b_weight.clone()))
        offset += width

    del concatenated, u, s, vh, sqrt_s
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return factors


def _compress_basis_sharing(model, state_dict, args, device: torch.device):
    layers = _target_layers(model)
    low_rank_modules: dict[str, dict[str, int]] = {}
    group_metadata: dict[str, list[list[int]]] = {}

    for module_name in TARGET_MODULES:
        groups = _basis_groups(len(layers), module_name, int(args.basis_group_size))
        short_name = module_name.rsplit(".", 1)[-1].replace("_proj", "")
        group_metadata[f"{short_name}_groups"] = groups
        for group in groups:
            modules = [layers[layer_idx].get_submodule(module_name) for layer_idx in group]
            first = modules[0]
            rank = _basis_rank_from_keep_ratio(
                first.in_features,
                first.out_features,
                len(group),
                float(args.keep_ratio),
            )
            print(f"[basis_sharing] {module_name} layers={group} rank={rank}", flush=True)
            factors = _factorize_basis_group(modules, rank, device, first.weight.dtype)
            for layer_idx, (a_weight, b_weight) in zip(group, factors, strict=True):
                module_key = _module_key(layer_idx, module_name)
                _replace_state_dict_entry(state_dict, module_key, a_weight, b_weight)
                low_rank_modules[module_key] = {"rank": int(rank)}

    metadata = {
        "basis_sharing_materialized": True,
        "basis_sharing_no_lora_finetune": True,
        "basis_sharing_group_size": int(args.basis_group_size),
        "basis_sharing_share_modules": sorted(SHARED_BASIS_MODULES),
        "basis_sharing_private_modules": sorted(PRIVATE_BASIS_MODULES),
        "basis_sharing_groups": group_metadata,
    }
    return low_rank_modules, metadata


def _save_artifact(model, tokenizer, state_dict, low_rank_modules, metadata, args) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.unsafe_overwrite:
        raise FileExistsError(f"{output_dir} already exists and is not empty; pass --unsafe-overwrite to replace files.")
    output_dir.mkdir(parents=True, exist_ok=True)

    method_label = "svdllm_v2" if args.method == "svdllm_v2" else "basis_sharing"
    config = _build_lowrank_config(model, low_rank_modules, method_label, metadata)
    config.save_pretrained(output_dir)
    _copy_model_code(output_dir)
    tokenizer.save_pretrained(output_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(output_dir)

    run_spec = {
        "model_id": args.model_id,
        "method": args.method,
        "keep_ratio": float(args.keep_ratio),
        "output_dir": str(output_dir),
        "target_modules": list(TARGET_MODULES),
        **metadata,
    }
    (output_dir / "run_spec.json").write_text(json.dumps(_jsonify(run_spec), indent=2, sort_keys=True) + "\n")

    print(f"[save] writing {len(state_dict)} tensors to {output_dir}", flush=True)
    save_torch_state_dict(
        state_dict,
        str(output_dir),
        max_shard_size=args.max_shard_size,
        safe_serialization=True,
    )
    print(f"[done] saved {args.method} keep_ratio={args.keep_ratio} to {output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Build Qwen3-8B LowRankArena ABLinear compression artifacts.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--method", required=True, choices=("svdllm_v2", "basis_sharing"))
    parser.add_argument("--keep-ratio", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--basis-group-size", type=int, default=2)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--unsafe-overwrite", action="store_true")
    args = parser.parse_args()
    if not (0.0 < float(args.keep_ratio) <= 1.0):
        raise ValueError(f"--keep-ratio must be in (0, 1], got {args.keep_ratio}")
    if int(args.basis_group_size) < 1:
        raise ValueError("--basis-group-size must be positive")
    if int(args.calibration_samples) < 1:
        raise ValueError("--calibration-samples must be positive")
    if int(args.sequence_length) < 1:
        raise ValueError("--sequence-length must be positive")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[load] model={args.model_id} method={args.method} keep_ratio={args.keep_ratio} device={device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    state_dict = model.state_dict()

    if args.method == "svdllm_v2":
        low_rank_modules, metadata = _compress_svdllm_v2(model, tokenizer, state_dict, args, device)
    elif args.method == "basis_sharing":
        low_rank_modules, metadata = _compress_basis_sharing(model, state_dict, args, device)
    else:  # pragma: no cover
        raise ValueError(args.method)

    _save_artifact(model, tokenizer, state_dict, low_rank_modules, metadata, args)


if __name__ == "__main__":
    main()
