#!/usr/bin/env python3
"""Normalize an AA-SVD native save_pretrained directory to ABLinear HF format."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import save_torch_state_dict
from safetensors import safe_open
from transformers import AutoTokenizer, LlamaTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
MODELING_DIR = REPO_ROOT / "src" / "modeling"
sys.path.insert(0, str(REPO_ROOT))

from src.modeling.llama.configuration_lowrank_llama import LowRankLlamaConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-ratio", type=float, required=True)
    parser.add_argument("--target-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--unsafe-overwrite", action="store_true")
    return parser.parse_args()


def load_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    shards = sorted(checkpoint.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors files under {checkpoint}")
    state = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in state:
                    raise ValueError(f"Duplicate state key {key}")
                state[key] = handle.get_tensor(key)
    return state


def normalize_state_dict(state, target_dtype):
    normalized = {}
    specs = {}
    shapes = {}
    for key, value in state.items():
        new_key = key
        if key.endswith(".U.weight"):
            module = key[:-len(".U.weight")]
            new_key = f"{module}.ALinear.weight"
            specs[module] = {"rank": int(value.shape[1])}
            shapes.setdefault(module, {})["a"] = tuple(value.shape)
        elif key.endswith(".V.weight"):
            module = key[:-len(".V.weight")]
            new_key = f"{module}.BLinear.weight"
            shapes.setdefault(module, {})["b"] = tuple(value.shape)
        elif key.endswith(".U.bias"):
            module = key[:-len(".U.bias")]
            new_key = f"{module}.ALinear.bias"
        elif key.endswith(".V.bias"):
            raise ValueError(f"Unexpected bias on AA-SVD input factor: {key}")
        if value.is_floating_point():
            value = value.to(target_dtype)
        normalized[new_key] = value.contiguous()

    dense_params = 0
    factor_params = 0
    for module, spec in specs.items():
        a_shape = shapes.get(module, {}).get("a")
        b_shape = shapes.get(module, {}).get("b")
        if a_shape is None or b_shape is None:
            raise ValueError(f"Incomplete AA-SVD factors for {module}: {shapes.get(module)}")
        rank = spec["rank"]
        if a_shape[1] != rank or b_shape[0] != rank:
            raise ValueError(f"Rank mismatch for {module}: A={a_shape}, B={b_shape}, rank={rank}")
        dense_params += a_shape[0] * b_shape[1]
        factor_params += a_shape[0] * rank + rank * b_shape[1]
    return normalized, specs, dense_params, factor_params


def copy_runtime(output_dir):
    for source in (
        MODELING_DIR / "common.py",
        MODELING_DIR / "llama" / "configuration_lowrank_llama.py",
        MODELING_DIR / "llama" / "modeling_lowrank_llama.py",
    ):
        shutil.copy2(source, output_dir / source.name)


def save_tokenizer(base_model, output_dir):
    """Load legacy LLaMA tokenizers when AutoTokenizer returns a false sentinel."""
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, local_files_only=True
    )
    if not hasattr(tokenizer, "save_pretrained"):
        tokenizer = LlamaTokenizer.from_pretrained(
            base_model, trust_remote_code=True, local_files_only=True
        )
    tokenizer.save_pretrained(output_dir)


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.unsafe_overwrite:
            raise FileExistsError(f"Non-empty output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    target_dtype = getattr(torch, args.target_dtype)
    state = load_state_dict(args.native_checkpoint.expanduser().resolve())
    state, specs, dense_params, factor_params = normalize_state_dict(state, target_dtype)

    config = LowRankLlamaConfig.from_pretrained(args.base_model)
    config.low_rank_modules = specs
    config.low_rank_method = "aa_svd"
    config.low_rank_schema = "ABLinear"
    config.low_rank_format_version = 1
    config.torch_dtype = args.target_dtype
    config.architectures = ["LowRankLlamaForCausalLM"]
    config.auto_map = {
        "AutoConfig": "configuration_lowrank_llama.LowRankLlamaConfig",
        "AutoModel": "modeling_lowrank_llama.LowRankLlamaModel",
        "AutoModelForCausalLM": "modeling_lowrank_llama.LowRankLlamaForCausalLM",
    }
    config.save_pretrained(output_dir)
    save_tokenizer(args.base_model, output_dir)
    copy_runtime(output_dir)
    save_torch_state_dict(
        state, str(output_dir), max_shard_size=args.max_shard_size, safe_serialization=True
    )
    metadata = {
        "method": "AA-SVD",
        "variant": "objective_2_plus_block_local_refinement",
        "base_model": args.base_model,
        "requested_keep_ratio": args.keep_ratio,
        "dense_target_parameters": dense_params,
        "factor_target_parameters": factor_params,
        "achieved_target_keep_ratio": factor_params / dense_params,
        "precision": args.target_dtype,
        "remapping": False,
        "quantization": False,
        "external_recovery": False,
        "intrinsic_local_optimization": "25-epoch block-local MSE refinement",
    }
    (output_dir / "lowrankarena_method.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
