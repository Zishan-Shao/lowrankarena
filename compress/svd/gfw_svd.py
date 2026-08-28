#!/usr/bin/env python3
"""Parameterised GFW-SVD to LowRankArena Hugging Face exporter.

This file is LowRankArena-owned adapter code. It imports the numerical
Kronecker-factor helpers from a pinned FisherKronecker checkout at runtime and
does not modify or present those upstream files as LowRankArena code.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELING_DIR = REPO_ROOT / "src" / "modeling"
TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
UPSTREAM_COMMIT = "d009b028c1e73545d8c604bcd29c1e091c8f341c"
_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "fp32": "float32",
    "float32": "float32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a FisherKronecker GFW-SVD artifact in LowRankArena HF format."
    )
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--keep-ratio", type=float, required=True)
    parser.add_argument("--kron-factors-dir", type=Path, required=True)
    parser.add_argument(
        "--rank-config",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping module names to keep ratios. Without it, "
            "--keep-ratio is applied uniformly to all LLaMA projection modules."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compute-device", default="cpu")
    parser.add_argument(
        "--target-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--rank-align", type=int, default=8)
    parser.add_argument("--min-rank", type=int, default=1)
    parser.add_argument("--reg-alpha", type=float, default=1e-1)
    parser.add_argument("--max-reg-tries", type=int, default=10)
    parser.add_argument("--alpha-increase-factor", type=float, default=1e-1)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--unsafe-overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _helper_path(upstream_root: Path) -> Path:
    return upstream_root / "llama" / "calibrate_llama_with_kronsvd.py"


def _require_packages() -> list[str]:
    missing: list[str] = []
    for package in (
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "safetensors",
        "huggingface_hub",
    ):
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    return missing


def _normalise_module_name(name: str) -> str:
    return name[: -len(".weight")] if name.endswith(".weight") else name


def load_rank_config(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--rank-config must contain a JSON object")

    result: dict[str, float] = {}
    for raw_name, raw_spec in payload.items():
        if isinstance(raw_spec, dict):
            if "keep_ratio" not in raw_spec:
                raise ValueError(f"{raw_name}: rank config object needs keep_ratio")
            raw_ratio = raw_spec["keep_ratio"]
        else:
            raw_ratio = raw_spec
        ratio = float(raw_ratio)
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"{raw_name}: keep ratio must be in (0, 1], got {ratio}")
        result[_normalise_module_name(str(raw_name))] = ratio
    return result


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    helper = _helper_path(args.upstream_root.expanduser().resolve())
    factors_dir = args.kron_factors_dir.expanduser().resolve()

    if not 0.0 < args.keep_ratio <= 1.0:
        errors.append("--keep-ratio must be in (0, 1]")
    if args.rank_align < 1:
        errors.append("--rank-align must be >= 1")
    if args.min_rank < 1:
        errors.append("--min-rank must be >= 1")
    if not helper.is_file():
        errors.append(f"upstream helper is missing: {helper}")
    if not factors_dir.is_dir():
        errors.append(f"Kronecker factor directory is missing: {factors_dir}")
        factor_count = 0
    else:
        factor_count = sum(1 for _ in factors_dir.glob("*.safetensors"))
        if factor_count == 0:
            errors.append(f"no .safetensors Kronecker factors found in {factors_dir}")

    missing_packages = _require_packages()
    if missing_packages:
        errors.append("missing Python packages: " + ", ".join(missing_packages))

    rank_config_count = 0
    if args.rank_config is not None:
        rank_path = args.rank_config.expanduser().resolve()
        if not rank_path.is_file():
            errors.append(f"rank config is missing: {rank_path}")
        else:
            try:
                rank_config_count = len(load_rank_config(rank_path))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                errors.append(f"invalid rank config: {exc}")
    else:
        warnings.append("using uniform keep ratio for all supported LLaMA projections")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "upstream_helper": str(helper),
        "upstream_commit": UPSTREAM_COMMIT,
        "kron_factor_count": factor_count,
        "rank_config_count": rank_config_count,
    }


def _load_upstream_helper(upstream_root: Path) -> ModuleType:
    path = _helper_path(upstream_root)
    spec = importlib.util.spec_from_file_location("lowrankarena_gfw_upstream", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import FisherKronecker helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in (
        "calculate_rank_from_ratio",
        "prepare_kron_svd_components",
        "build_factored_layer_from_components",
    ):
        if not hasattr(module, name):
            raise AttributeError(f"Pinned upstream helper does not define {name}")
    return module


def _copy_runtime(output_dir: Path) -> None:
    for source in (
        MODELING_DIR / "common.py",
        MODELING_DIR / "llama" / "configuration_lowrank_llama.py",
        MODELING_DIR / "llama" / "modeling_lowrank_llama.py",
    ):
        shutil.copy2(source, output_dir / source.name)


def _prepare_output(output_dir: Path, unsafe_overwrite: bool) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not unsafe_overwrite:
            raise FileExistsError(f"Non-empty output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def export(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from huggingface_hub import save_torch_state_dict
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.modeling.llama.configuration_lowrank_llama import LowRankLlamaConfig

    helper = _load_upstream_helper(args.upstream_root.expanduser().resolve())
    output_dir = _prepare_output(args.output_dir, args.unsafe_overwrite)
    factors_dir = args.kron_factors_dir.expanduser().resolve()
    rank_config = load_rank_config(args.rank_config)
    target_dtype = getattr(torch, args.target_dtype)
    compute_device = torch.device(args.compute_device)

    dense_config = AutoConfig.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )
    if dense_config.model_type != "llama":
        raise ValueError(
            f"GFW-SVD adapter currently supports LLaMA only, got {dense_config.model_type}"
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=target_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.endswith(TARGET_SUFFIXES)
    }
    if not modules:
        raise ValueError("No supported LLaMA projection modules were found")

    unknown_config = sorted(set(rank_config) - set(modules))
    if unknown_config:
        raise ValueError(
            "Rank config contains unknown/non-target modules: "
            + ", ".join(unknown_config[:20])
        )

    ratios = {name: rank_config.get(name, args.keep_ratio) for name in modules}
    required_factors = {
        name: factors_dir / f"{name.replace('.', '_')}.safetensors"
        for name, ratio in ratios.items()
        if ratio < 1.0
    }
    missing_factors = [name for name, path in required_factors.items() if not path.is_file()]
    if missing_factors:
        raise FileNotFoundError(
            f"Missing Kronecker factors for {len(missing_factors)} target modules; "
            f"first missing: {missing_factors[0]}"
        )

    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    specs: dict[str, dict[str, int]] = {}
    dense_target_parameters = 0
    factor_target_parameters = 0

    for name, module in modules.items():
        ratio = ratios[name]
        if ratio >= 1.0:
            continue
        factor_payload = load_file(required_factors[name], device="cpu")
        if "XF" not in factor_payload or "YF" not in factor_payload:
            raise KeyError(f"{required_factors[name]} must contain XF and YF")
        xf = factor_payload["XF"].to(dtype=torch.float32).numpy()
        yf = factor_payload["YF"].to(dtype=torch.float32).numpy()
        components = helper.prepare_kron_svd_components(
            xf,
            yf,
            module.out_features,
            module.in_features,
            args.reg_alpha,
            args.max_reg_tries,
            args.alpha_increase_factor,
        )
        if components is None:
            raise RuntimeError(f"Failed to prepare Kronecker components for {name}")
        rank = helper.calculate_rank_from_ratio(
            module.in_features,
            module.out_features,
            ratio,
            args.rank_align,
            args.min_rank,
        )
        factorized = helper.build_factored_layer_from_components(
            module,
            components,
            rank,
            compute_device,
            target_dtype,
        )
        if factorized is None:
            raise RuntimeError(f"GFW-SVD factorization failed for {name}")

        weight_key = f"{name}.weight"
        if weight_key not in state:
            raise KeyError(f"Dense state dict is missing {weight_key}")
        dense_weight = state.pop(weight_key)
        state[f"{name}.BLinear.weight"] = (
            factorized[0].weight.detach().to("cpu", dtype=target_dtype).contiguous()
        )
        state[f"{name}.ALinear.weight"] = (
            factorized[1].weight.detach().to("cpu", dtype=target_dtype).contiguous()
        )
        bias_key = f"{name}.bias"
        if bias_key in state:
            state[f"{name}.ALinear.bias"] = state.pop(bias_key).to(target_dtype)

        specs[name] = {"rank": int(rank), "keep_ratio": float(ratio)}
        dense_target_parameters += dense_weight.numel()
        factor_target_parameters += (
            factorized[0].weight.numel() + factorized[1].weight.numel()
        )
        del factor_payload, xf, yf, components, factorized, dense_weight

    if not specs:
        raise ValueError("No modules were compressed; choose --keep-ratio below 1")
    state = {
        key: (value.to(target_dtype) if value.is_floating_point() else value).contiguous()
        for key, value in state.items()
    }
    del model
    if compute_device.type == "cuda":
        torch.cuda.empty_cache()

    config = LowRankLlamaConfig.from_pretrained(args.model, revision=args.revision)
    config.low_rank_modules = specs
    config.low_rank_method = "gfw_svd"
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
    AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        use_fast=False,
    ).save_pretrained(output_dir)
    _copy_runtime(output_dir)
    save_torch_state_dict(
        state,
        str(output_dir),
        max_shard_size=args.max_shard_size,
        safe_serialization=True,
    )

    achieved_ratio = factor_target_parameters / dense_target_parameters
    metadata = {
        "method": "GFW-SVD",
        "base_model": args.model,
        "base_revision": args.revision,
        "upstream_repository": "https://github.com/sayankotor/FisherKronecker",
        "upstream_commit": UPSTREAM_COMMIT,
        "requested_keep_ratio": args.keep_ratio,
        "rank_allocation": "explicit" if rank_config else "uniform",
        "rank_alignment": args.rank_align,
        "precision": args.target_dtype,
        "compressed_module_count": len(specs),
        "dense_target_parameters": dense_target_parameters,
        "factor_target_parameters": factor_target_parameters,
        "achieved_target_keep_ratio": achieved_ratio,
        "kron_factors_dir": str(factors_dir),
    }
    (output_dir / "lowrankarena_method.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"output_dir": str(output_dir), **metadata}


def _build_command(request: Any, baseline_path: str, output_dir: Path) -> list[str]:
    dtype = _DTYPE_ALIASES.get(request.precision.lower())
    if dtype is None:
        raise ValueError(
            f"Unsupported GFW-SVD precision '{request.precision}'; "
            f"choose one of {sorted(_DTYPE_ALIASES)}"
        )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--upstream-root",
        baseline_path,
        "--model",
        request.model,
        "--revision",
        request.revision,
        "--keep-ratio",
        str(request.ratio),
        "--kron-factors-dir",
        str(request.extra.get("kron_factors_dir", "<required:kron_factors_dir>")),
        "--output-dir",
        str(output_dir.resolve()),
        "--compute-device",
        str(request.extra.get("device", "cpu")),
        "--target-dtype",
        dtype,
        "--rank-align",
        str(request.extra.get("rank_align", 8)),
        "--min-rank",
        str(request.extra.get("min_rank", 1)),
        "--max-shard-size",
        str(request.extra.get("max_shard_size", "5GB")),
    ]
    if request.extra.get("rank_config"):
        command.extend(["--rank-config", str(request.extra["rank_config"])])
    if str(request.extra.get("unsafe_overwrite", "false")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        command.append("--unsafe-overwrite")
    return command


def build(request):
    """Build or execute GFW-SVD through the unified compression contract."""
    from compress.common import prepare_baseline
    from compress.save import execute_artifact, save_artifact

    baseline = prepare_baseline(request)
    if baseline.path is None:
        raise RuntimeError(
            "GFW-SVD source is unavailable. Use --clone-baseline or restore the "
            "curated compress/svd/GFW-SVD snapshot."
        )
    output_dir = request.artifact_root / request.artifact_id / "weights"
    artifact = save_artifact(
        request,
        baseline=baseline,
        command=_build_command(request, baseline.path, output_dir),
        output_format="lowrankarena_hf",
        status="planned",
        ready_for_load=False,
        notes=(
            "GFW-SVD using pinned FisherKronecker factorization helpers and "
            "LowRankArena's loadable ABLinear checkpoint schema."
        ),
        register=False,
    )
    return execute_artifact(request, artifact, baseline=baseline) if request.execute else artifact


def main() -> None:
    args = parse_args()
    report = preflight(args)
    if args.preflight_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["ok"] else 2)
    if not report["ok"]:
        raise SystemExit(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(export(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
