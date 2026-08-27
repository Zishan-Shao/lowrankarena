#!/usr/bin/env python3
"""Export a Zero-Sum-SVD pickle checkpoint as a uniform-precision HF artifact."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import save_torch_state_dict


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
MODELING_DIR = REPO_ROOT / "src" / "modeling"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.modeling.llama.configuration_lowrank_llama import LowRankLlamaConfig
from utils.model_utils import LowRankLinear


TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-ratio", type=float, required=True)
    parser.add_argument("--target-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--unsafe-overwrite", action="store_true")
    return parser.parse_args()


def copy_runtime(output_dir):
    for source in (
        MODELING_DIR / "common.py",
        MODELING_DIR / "llama" / "configuration_lowrank_llama.py",
        MODELING_DIR / "llama" / "modeling_lowrank_llama.py",
    ):
        shutil.copy2(source, output_dir / source.name)


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.unsafe_overwrite:
            raise FileExistsError(f"Non-empty output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    payload = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    model = payload["model"]
    tokenizer = payload.get("tokenizer")
    target_dtype = getattr(torch, args.target_dtype)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    specs = {}
    dense_params = 0
    retained_params = 0

    for module_name, module in model.named_modules():
        if not module_name.endswith(TARGET_SUFFIXES):
            continue
        if isinstance(module, LowRankLinear):
            in_features = int(module.in_features)
            out_features = int(module.out_features)
            dense_params += in_features * out_features
            retained_params += int(module.rank) * (in_features + out_features)
        elif isinstance(module, torch.nn.Linear):
            # Zero-Sum-SVD allocates a global budget.  Modules whose selected
            # rank reaches the dense boundary remain ordinary Linear layers;
            # they still belong in the benchmark's target-parameter budget.
            dense_params += int(module.in_features) * int(module.out_features)
            retained_params += int(module.in_features) * int(module.out_features)
            continue
        else:
            raise TypeError(f"Unsupported target module {module_name}: {type(module).__name__}")

        specs[module_name] = {"rank": int(module.rank)}
        a_old = f"{module_name}.u_proj.weight"
        b_old = f"{module_name}.v_proj.weight"
        state[f"{module_name}.ALinear.weight"] = state.pop(a_old)
        state[f"{module_name}.BLinear.weight"] = state.pop(b_old)
        old_bias = f"{module_name}.u_proj.bias"
        if old_bias in state:
            state[f"{module_name}.ALinear.bias"] = state.pop(old_bias)
        if f"{module_name}.v_proj.bias" in state:
            raise ValueError(f"Unexpected input-factor bias for {module_name}")

    if not specs:
        raise ValueError("Checkpoint contains no Zero-Sum-SVD LowRankLinear modules")
    state = {
        key: (value.to(target_dtype) if value.is_floating_point() else value).contiguous()
        for key, value in state.items()
    }

    config = LowRankLlamaConfig.from_dict(model.config.to_dict())
    config.low_rank_modules = specs
    config.low_rank_method = "zero_sum_svd"
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
    if tokenizer is None:
        raise ValueError("Checkpoint does not contain a tokenizer")
    tokenizer.save_pretrained(output_dir)
    copy_runtime(output_dir)
    save_torch_state_dict(
        state, str(output_dir), max_shard_size=args.max_shard_size, safe_serialization=True
    )
    metadata = {
        "method": "ZS-SVD",
        "variant": "zero_sum_one_shot",
        "base_model": payload.get("model_name", getattr(model.config, "_name_or_path", None)),
        "requested_keep_ratio": args.keep_ratio,
        "dense_target_parameters": dense_params,
        "retained_target_parameters": retained_params,
        "achieved_target_keep_ratio": retained_params / dense_params,
        "precision": args.target_dtype,
        "remapping": False,
        "quantization": False,
        "external_recovery": False,
        "intrinsic_correction_steps": 0,
    }
    (output_dir / "lowrankarena_method.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
