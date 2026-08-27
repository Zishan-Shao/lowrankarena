#!/usr/bin/env python3
"""Export Swift-SVD output-space projections as a standard low-rank HF artifact.

The upstream quality evaluator reconstructs ``W' = V_r V_r^T W`` into a dense
matrix.  This exporter stores the algebraically identical factors
``A = V_r`` and ``B = V_r^T W`` so the physical parameter count matches the
requested keep-ratio and the artifact can use LowRankArena's generic runtime.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import save_torch_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
MODELING_DIR = REPO_ROOT / "src" / "modeling"
sys.path.insert(0, str(SCRIPT_DIR))  # makes utils.svd importable while unpickling
sys.path.insert(0, str(REPO_ROOT))

from src.modeling.llama.configuration_lowrank_llama import LowRankLlamaConfig


MODULE_SUFFIXES = {
    "query": "self_attn.q_proj",
    "key": "self_attn.k_proj",
    "value": "self_attn.v_proj",
    "output": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--svd-file", type=Path, required=True)
    parser.add_argument("--rank-allocation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-ratio", type=float, required=True)
    parser.add_argument("--compute-device", default="cpu")
    parser.add_argument("--target-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--unsafe-overwrite", action="store_true")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _copy_runtime(output_dir: Path) -> None:
    for source in (
        MODELING_DIR / "common.py",
        MODELING_DIR / "llama" / "configuration_lowrank_llama.py",
        MODELING_DIR / "llama" / "modeling_lowrank_llama.py",
    ):
        shutil.copy2(source, output_dir / source.name)


def _rank_map(allocation: list[dict]) -> dict[str, dict[str, int]]:
    specs: dict[str, dict[str, int]] = {}
    for layer_idx, row in enumerate(allocation):
        for swift_name, suffix in MODULE_SUFFIXES.items():
            specs[f"model.layers.{layer_idx}.{suffix}"] = {
                "rank": int(row[f"{swift_name}_rank"])
            }
    return specs


@torch.inference_mode()
def build_factorized_state_dict(model, svd_list, allocation, compute_device, target_dtype):
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    specs = _rank_map(allocation)
    dense_target_params = 0
    factor_target_params = 0

    for layer_idx, row in enumerate(allocation):
        for swift_name, suffix in MODULE_SUFFIXES.items():
            module_name = f"model.layers.{layer_idx}.{suffix}"
            weight_key = f"{module_name}.weight"
            if weight_key not in state:
                raise KeyError(f"Missing base-model weight: {weight_key}")

            weight = state.pop(weight_key).to(compute_device, dtype=torch.float32)
            basis = svd_list[layer_idx][swift_name].V
            rank = int(row[f"{swift_name}_rank"])
            if basis.shape[0] != weight.shape[0]:
                raise ValueError(
                    f"{module_name}: output basis {tuple(basis.shape)} does not match weight {tuple(weight.shape)}"
                )
            if rank > min(weight.shape[0], weight.shape[1], basis.shape[1]):
                raise ValueError(f"{module_name}: invalid rank {rank} for weight {tuple(weight.shape)}")

            a = basis[:, :rank].to(compute_device, dtype=torch.float32)
            b = a.T @ weight
            state[f"{module_name}.ALinear.weight"] = a.to("cpu", dtype=target_dtype).contiguous()
            state[f"{module_name}.BLinear.weight"] = b.to("cpu", dtype=target_dtype).contiguous()

            bias_key = f"{module_name}.bias"
            if bias_key in state:
                state[f"{module_name}.ALinear.bias"] = state.pop(bias_key).to(target_dtype)

            dense_target_params += weight.numel()
            factor_target_params += a.numel() + b.numel()
            del weight, a, b

    state = {
        key: (value.to(target_dtype) if value.is_floating_point() else value).contiguous()
        for key, value in state.items()
    }
    budget = {
        "dense_target_parameters": dense_target_params,
        "factor_target_parameters": factor_target_params,
        "achieved_target_keep_ratio": factor_target_params / dense_target_params,
    }
    return state, specs, budget


def main() -> None:
    args = parse_args()
    if not 0.0 < args.keep_ratio <= 1.0:
        raise ValueError("--keep-ratio must be in (0, 1]")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.unsafe_overwrite:
            raise FileExistsError(f"Non-empty output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    target_dtype = _dtype(args.target_dtype)
    compute_device = torch.device(args.compute_device)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=target_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if model.config.model_type != "llama":
        raise ValueError(f"This exporter currently supports LLaMA artifacts, got {model.config.model_type}")

    with args.svd_file.open("rb") as handle:
        svd_list = pickle.load(handle)
    with args.rank_allocation.open("rb") as handle:
        allocation = pickle.load(handle)
    if len(svd_list) != len(model.model.layers) or len(allocation) != len(model.model.layers):
        raise ValueError("SVD/rank allocation layer count does not match the base model")

    state, specs, budget = build_factorized_state_dict(
        model, svd_list, allocation, compute_device, target_dtype
    )
    del model
    if compute_device.type == "cuda":
        torch.cuda.empty_cache()

    base_config = LowRankLlamaConfig.from_pretrained(args.base_model)
    base_config.low_rank_modules = specs
    base_config.low_rank_method = "swift_svd_uniform"
    base_config.low_rank_schema = "ABLinear"
    base_config.low_rank_format_version = 1
    base_config.torch_dtype = args.target_dtype
    base_config.architectures = ["LowRankLlamaForCausalLM"]
    base_config.auto_map = {
        "AutoConfig": "configuration_lowrank_llama.LowRankLlamaConfig",
        "AutoModel": "modeling_lowrank_llama.LowRankLlamaModel",
        "AutoModelForCausalLM": "modeling_lowrank_llama.LowRankLlamaForCausalLM",
    }
    base_config.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, use_fast=False
    ).save_pretrained(output_dir)
    _copy_runtime(output_dir)
    save_torch_state_dict(
        state,
        str(output_dir),
        max_shard_size=args.max_shard_size,
        safe_serialization=True,
    )

    metadata = {
        "method": "Swift-SVD",
        "variant": "uniform",
        "base_model": args.base_model,
        "requested_keep_ratio": args.keep_ratio,
        "precision": args.target_dtype,
        "remapping": False,
        "quantization": False,
        "recovery": False,
        **budget,
    }
    (output_dir / "lowrankarena_method.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
