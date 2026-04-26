#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import HfApi  # noqa: E402
from scripts.hf_repo_utils import infer_path_in_repo  # noqa: E402


def _patch_transformers_id_tensor_storage() -> None:
    # Some environments ship torch without torch.distributed.tensor.DTensor.
    # transformers.pytorch_utils.id_tensor_storage imports it unguarded.
    try:
        from transformers import pytorch_utils as _ptu  # type: ignore
    except Exception:
        return
    try:
        from transformers import modeling_utils as _mu  # type: ignore
    except Exception:
        _mu = None

    def _safe_id_tensor_storage(tensor: torch.Tensor) -> int:
        try:
            from torch.distributed.tensor import DTensor  # type: ignore
            if isinstance(tensor, DTensor):
                tensor = tensor.to_local()
        except Exception:
            pass
        try:
            storage_ptr = tensor.untyped_storage().data_ptr()
        except Exception:
            storage_ptr = tensor.storage().data_ptr()
        return storage_ptr

    _ptu.id_tensor_storage = _safe_id_tensor_storage
    if _mu is not None and hasattr(_mu, "id_tensor_storage"):
        _mu.id_tensor_storage = _safe_id_tensor_storage
        if not hasattr(_mu, "DTensor"):
            class _DummyDTensor:  # pragma: no cover
                pass
            _mu.DTensor = _DummyDTensor


LLAMA_CONFIGURATION_CODE = r'''from __future__ import annotations

from transformers.models.llama.configuration_llama import LlamaConfig


class DobiSVDLlamaConfig(LlamaConfig):
    model_type = "llama"
'''


QWEN3_CONFIGURATION_CODE = r'''from __future__ import annotations

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class DobiSVDQwen3Config(Qwen3Config):
    model_type = "qwen3"
'''


LLAMA_MODELING_CODE = r'''from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel
from .configuration_dobisvd_llama import DobiSVDLlamaConfig


class DobiSVDLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False):
        super().__init__()
        self.ALinear = nn.Linear(in_features, rank, bias=bias)
        self.BLinear = nn.Linear(rank, out_features, bias=False)

    def forward(self, x):
        return self.BLinear(self.ALinear(x))


def _resolve_parent(root: nn.Module, name: str):
    if "." not in name:
        return root, name
    parent_name, attr = name.rsplit(".", 1)
    parent = dict(root.named_modules())[parent_name]
    return parent, attr


def _apply_dobi_replacements(root: nn.Module, module_ranks: dict[str, int]):
    for name, rank in module_ranks.items():
        candidate_names = [name]
        if name.startswith("model."):
            candidate_names.append(name[len("model."):])
        resolved = None
        for candidate in candidate_names:
            try:
                resolved = _resolve_parent(root, candidate)
                break
            except KeyError:
                continue
        if resolved is None:
            continue
        parent, attr = resolved
        original = getattr(parent, attr)
        if not isinstance(original, nn.Linear):
            continue
        replacement = DobiSVDLinear(
            in_features=original.in_features,
            out_features=original.out_features,
            rank=int(rank),
            bias=original.bias is not None,
        )
        setattr(parent, attr, replacement)


class DobiSVDLlamaModel(LlamaModel):
    config_class = DobiSVDLlamaConfig
    _supports_attention_backend = True

    def __init__(self, config):
        super().__init__(config)
        module_ranks = getattr(config, "dobi_target_modules", {}) or {}
        _apply_dobi_replacements(self, module_ranks)


class DobiSVDLlamaForCausalLM(LlamaForCausalLM):
    config_class = DobiSVDLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = DobiSVDLlamaModel(config)
'''


QWEN3_MODELING_CODE = r'''from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model
from .configuration_dobisvd_qwen3 import DobiSVDQwen3Config


class DobiSVDLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False):
        super().__init__()
        self.ALinear = nn.Linear(in_features, rank, bias=bias)
        self.BLinear = nn.Linear(rank, out_features, bias=False)

    def forward(self, x):
        return self.BLinear(self.ALinear(x))


def _resolve_parent(root: nn.Module, name: str):
    if "." not in name:
        return root, name
    parent_name, attr = name.rsplit(".", 1)
    parent = dict(root.named_modules())[parent_name]
    return parent, attr


def _apply_dobi_replacements(root: nn.Module, module_ranks: dict[str, int]):
    for name, rank in module_ranks.items():
        candidate_names = [name]
        if name.startswith("model."):
            candidate_names.append(name[len("model."):])
        resolved = None
        for candidate in candidate_names:
            try:
                resolved = _resolve_parent(root, candidate)
                break
            except KeyError:
                continue
        if resolved is None:
            continue
        parent, attr = resolved
        original = getattr(parent, attr)
        if not isinstance(original, nn.Linear):
            continue
        replacement = DobiSVDLinear(
            in_features=original.in_features,
            out_features=original.out_features,
            rank=int(rank),
            bias=original.bias is not None,
        )
        setattr(parent, attr, replacement)


class DobiSVDQwen3Model(Qwen3Model):
    config_class = DobiSVDQwen3Config
    _supports_attention_backend = True

    def __init__(self, config):
        super().__init__(config)
        module_ranks = getattr(config, "dobi_target_modules", {}) or {}
        _apply_dobi_replacements(self, module_ranks)


class DobiSVDQwen3ForCausalLM(Qwen3ForCausalLM):
    config_class = DobiSVDQwen3Config

    def __init__(self, config):
        super().__init__(config)
        self.model = DobiSVDQwen3Model(config)
'''


def _load_bundle(bundle_path: Path):
    try:
        return torch.load(str(bundle_path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(bundle_path), map_location="cpu")


def _is_dobi_svd_layer(module) -> bool:
    cls_name = module.__class__.__name__
    if cls_name in {"SVDTransformLayer", "SVDTransformLayer_remapping"}:
        return True
    return hasattr(module, "ALinear") and hasattr(module, "BLinear")


def _collect_llama_target_modules(model) -> dict[str, int]:
    target_modules: dict[str, int] = {}
    for name, module in model.named_modules():
        if not name:
            continue
        if not _is_dobi_svd_layer(module):
            continue
        rank = int(module.ALinear.weight.shape[0])
        target_modules[name] = rank
    return target_modules


def _write_readme(out_dir: Path, bundle_path: Path, replaced: int) -> None:
    ratio = "unknown"
    stem = bundle_path.stem
    if "_" in stem:
        ratio = stem.split("_")[-1]
    text = (
        f"# DobiSVD Low-Rank HF Export (ratio {ratio})\n\n"
        "This folder keeps DobiSVD layers in low-rank form and is intended for "
        "Transformers loading with `trust_remote_code=True`.\n\n"
        "Contents:\n"
        "- `model.safetensors`: low-rank state dict\n"
        "- `modeling_dobisvd_llama.py`: custom loader that rebuilds low-rank layers\n"
        "- tokenizer/config files for `AutoModelForCausalLM.from_pretrained`\n\n"
        "Load example:\n"
        "```python\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "path = \".\"\n"
        "tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)\n"
        "model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)\n"
        "```\n"
    )
    (out_dir / "README.md").write_text(text, encoding="utf-8")

    timing = {
        "method": "DobiSVD",
        "export_variant": "low_rank_hf",
        "source_bundle": str(bundle_path),
        "replaced_layers": int(replaced),
        "exported_by": os.environ.get("USER", "unknown"),
        "exported_at": datetime.now().strftime("%Y-%m-%d"),
    }
    (out_dir / "dobi_lowrank_build_timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")


def _json_safe(value):
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _rewrite_config_json(out_dir: Path, config, target_modules: dict[str, int], model_type: str) -> None:
    config_path = out_dir / "config.json"
    source_config_path = Path(getattr(config, "_name_or_path", "")) / "config.json"
    if source_config_path.exists():
        payload = json.loads(source_config_path.read_text(encoding="utf-8"))
    else:
        payload = _json_safe(config.to_dict())
    if model_type == "llama":
        payload["architectures"] = ["DobiSVDLlamaForCausalLM"]
        payload["auto_map"] = {
            "AutoConfig": "configuration_dobisvd_llama.DobiSVDLlamaConfig",
            "AutoModel": "modeling_dobisvd_llama.DobiSVDLlamaModel",
            "AutoModelForCausalLM": "modeling_dobisvd_llama.DobiSVDLlamaForCausalLM",
        }
    elif model_type == "qwen3":
        payload["architectures"] = ["DobiSVDQwen3ForCausalLM"]
        payload["auto_map"] = {
            "AutoConfig": "configuration_dobisvd_qwen3.DobiSVDQwen3Config",
            "AutoModel": "modeling_dobisvd_qwen3.DobiSVDQwen3Model",
            "AutoModelForCausalLM": "modeling_dobisvd_qwen3.DobiSVDQwen3ForCausalLM",
        }
    else:
        raise RuntimeError(f"Unsupported model_type for config rewrite: {model_type}")
    payload["dobi_target_modules"] = target_modules
    config_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _normalize_config_for_save(model) -> None:
    torch_dtype = getattr(model.config, "torch_dtype", None)
    if isinstance(torch_dtype, torch.dtype):
        model.config.torch_dtype = str(torch_dtype).replace("torch.", "")


def _save_weights_only(model, out_dir: Path, max_shard_size: str) -> None:
    # Newer transformers can re-inject a non-JSON-serializable torch.dtype into
    # config serialization even after the config object is normalized. Save the
    # weights first, then let this script write a cleaned config.json.
    original_save_pretrained = model.config.save_pretrained
    try:
        model.config.save_pretrained = lambda *args, **kwargs: None  # type: ignore[method-assign]
        model.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size=max_shard_size)
    finally:
        model.config.save_pretrained = original_save_pretrained  # type: ignore[method-assign]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a DobiSVD bundle to a low-rank HF directory.")
    ap.add_argument("--bundle", required=True, help="Path to DobiSVD .pt bundle")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--max-shard-size", default="5GB", help="Max shard size for save_pretrained")
    ap.add_argument("--repo-id", default="", help="HF repo id, e.g. Duke-CEI-SVD/LowRankArena")
    ap.add_argument("--path-in-repo", default="", help="Path inside repo for upload; if omitted, infer canonical Dobi path")
    ap.add_argument("--upload", action="store_true", help="Upload output dir to Hugging Face")
    args = ap.parse_args()

    bundle_path = Path(args.bundle).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_bundle(bundle_path)
    if not isinstance(payload, dict) or "model" not in payload or "tokenizer" not in payload:
        raise RuntimeError("Bundle must be a dict with keys 'model' and 'tokenizer'.")
    model = payload["model"]
    tokenizer = payload["tokenizer"]

    model_type = getattr(model.config, "model_type", None)
    if model_type not in {"llama", "qwen3"}:
        raise RuntimeError(f"Only llama and qwen3 exports are implemented in this script, got: {model_type}")

    target_modules = _collect_llama_target_modules(model)
    if not target_modules:
        raise RuntimeError("No Dobi low-rank modules found in bundle.")

    if model_type == "llama":
        model.config.architectures = ["DobiSVDLlamaForCausalLM"]
        model.config.auto_map = {
            "AutoConfig": "configuration_dobisvd_llama.DobiSVDLlamaConfig",
            "AutoModel": "modeling_dobisvd_llama.DobiSVDLlamaModel",
            "AutoModelForCausalLM": "modeling_dobisvd_llama.DobiSVDLlamaForCausalLM",
        }
    else:
        model.config.architectures = ["DobiSVDQwen3ForCausalLM"]
        model.config.auto_map = {
            "AutoConfig": "configuration_dobisvd_qwen3.DobiSVDQwen3Config",
            "AutoModel": "modeling_dobisvd_qwen3.DobiSVDQwen3Model",
            "AutoModelForCausalLM": "modeling_dobisvd_qwen3.DobiSVDQwen3ForCausalLM",
        }
    model.config.dobi_target_modules = target_modules

    _patch_transformers_id_tensor_storage()
    _normalize_config_for_save(model)
    _save_weights_only(model, out_dir=out_dir, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(str(out_dir))
    if model_type == "llama":
        (out_dir / "configuration_dobisvd_llama.py").write_text(LLAMA_CONFIGURATION_CODE, encoding="utf-8")
        (out_dir / "modeling_dobisvd_llama.py").write_text(LLAMA_MODELING_CODE, encoding="utf-8")
    else:
        (out_dir / "configuration_dobisvd_qwen3.py").write_text(QWEN3_CONFIGURATION_CODE, encoding="utf-8")
        (out_dir / "modeling_dobisvd_qwen3.py").write_text(QWEN3_MODELING_CODE, encoding="utf-8")
    _rewrite_config_json(out_dir=out_dir, config=model.config, target_modules=target_modules, model_type=model_type)
    _write_readme(out_dir=out_dir, bundle_path=bundle_path, replaced=len(target_modules))
    print(f"[export] wrote low-rank HF files to: {out_dir}")

    if args.upload:
        if not args.repo_id:
            raise ValueError("--upload requires --repo-id")
        path_in_repo = infer_path_in_repo(bundle_path=bundle_path, explicit_path_in_repo=args.path_in_repo, default_method="DobiSVD")
        api = HfApi()
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=args.repo_id,
            repo_type="model",
            path_in_repo=path_in_repo,
            commit_message=f"Add DobiSVD low-rank HF export: {path_in_repo}",
        )
        print(f"[upload] done -> https://huggingface.co/{args.repo_id}/tree/main/{path_in_repo}")


if __name__ == "__main__":
    main()
