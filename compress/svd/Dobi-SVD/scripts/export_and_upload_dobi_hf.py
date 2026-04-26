#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hf_repo_utils import infer_path_in_repo


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


def _load_bundle(bundle_path: Path):
    try:
        return torch.load(str(bundle_path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(bundle_path), map_location="cpu")


def _resolve_parent(root: nn.Module, name: str) -> Tuple[nn.Module, str]:
    if "." not in name:
        return root, name
    parent_name, attr = name.rsplit(".", 1)
    parent = dict(root.named_modules())[parent_name]
    return parent, attr


def _is_dobi_svd_layer(module: nn.Module) -> bool:
    cls_name = module.__class__.__name__
    if cls_name in {"SVDTransformLayer", "SVDTransformLayer_remapping"}:
        return True
    return hasattr(module, "ALinear") and hasattr(module, "BLinear")


@torch.no_grad()
def _fuse_to_linear(module: nn.Module) -> nn.Linear:
    a = module.ALinear.weight.detach().to(torch.float32)  # [rank, in_features]
    b = module.BLinear.weight.detach().to(torch.float32)  # [out_features, rank]
    out_features = int(b.shape[0])
    in_features = int(a.shape[1])
    has_bias = getattr(module.BLinear, "bias", None) is not None

    fused = b @ a  # [out_features, in_features]
    target_dtype = module.BLinear.weight.dtype if module.BLinear.weight.is_floating_point() else torch.float32
    linear = nn.Linear(in_features, out_features, bias=has_bias)
    linear.weight.copy_(fused.to(dtype=target_dtype))
    if has_bias:
        linear.bias.copy_(module.BLinear.bias.detach().to(dtype=target_dtype))
    return linear


@torch.no_grad()
def fuse_model_inplace(model: nn.Module) -> int:
    replace_names = [name for name, mod in model.named_modules() if name and _is_dobi_svd_layer(mod)]
    replaced = 0
    for name in replace_names:
        module = dict(model.named_modules())[name]
        parent, attr = _resolve_parent(model, name)
        fused_linear = _fuse_to_linear(module)
        setattr(parent, attr, fused_linear)
        replaced += 1
        del module
        gc.collect()
    return replaced


def write_readme(out_dir: Path, bundle_path: Path, replaced: int) -> None:
    ratio = "unknown"
    stem = bundle_path.stem
    if "_" in stem:
        ratio = stem.split("_")[-1]
    title = f"DobiSVD Qwen3-8B-Base (ratio {ratio})"
    text = (
        f"# {title}\n\n"
        "This folder is prepared for Hugging Face upload.\n\n"
        "Contents:\n"
        "- `model.pt`: original Dobi bundle (`{'model','tokenizer'}`)\n"
        "- `model-*.safetensors` + `model.safetensors.index.json`: HF-compatible weights\n"
        "- tokenizer/config files for direct `from_pretrained`\n\n"
        "Load example:\n"
        "```python\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "path = \".\"\n"
        "tok = AutoTokenizer.from_pretrained(path)\n"
        "model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=\"auto\")\n"
        "```\n"
    )
    (out_dir / "README.md").write_text(text, encoding="utf-8")

    timing = {
        "method": "DobiSVD",
        "source_bundle": str(bundle_path),
        "export_note": "Fused SVDTransformLayer back to standard Linear for HF-compatible safetensors export.",
        "replaced_layers": int(replaced),
        "exported_by": os.environ.get("USER", "unknown"),
        "exported_at": datetime.now().strftime("%Y-%m-%d"),
    }
    (out_dir / "dobi_build_timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export DobiSVD .pt bundle to HF-compatible folder and optionally upload.")
    ap.add_argument("--bundle", required=True, help="Path to DobiSVD bundle .pt")
    ap.add_argument("--out-dir", required=True, help="Output directory for HF-ready files")
    ap.add_argument("--max-shard-size", default="5GB", help="Max shard size for save_pretrained")
    ap.add_argument("--repo-id", default="", help="HF repo id, e.g. Duke-CEI-SVD/LowRankArena")
    ap.add_argument("--path-in-repo", default="", help="Path inside repo for upload; if omitted, infer canonical Dobi path")
    ap.add_argument("--upload", action="store_true", help="Upload out-dir to HF")
    args = ap.parse_args()

    bundle_path = Path(args.bundle).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_bundle(bundle_path)
    if not isinstance(payload, dict) or "model" not in payload or "tokenizer" not in payload:
        raise RuntimeError("Bundle must be a dict with keys 'model' and 'tokenizer'.")
    model = payload["model"]
    tokenizer = payload["tokenizer"]

    replaced = fuse_model_inplace(model)
    print(f"[export] fused SVD layers: {replaced}")
    _patch_transformers_id_tensor_storage()

    model.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(str(out_dir))
    # Keep model.pt as the original low-rank Dobi bundle instead of fused dense model.
    shutil.copy2(str(bundle_path), str(out_dir / "model.pt"))
    write_readme(out_dir=out_dir, bundle_path=bundle_path, replaced=replaced)
    print(f"[export] wrote HF-ready files to: {out_dir}")

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
            commit_message=f"Add DobiSVD export: {path_in_repo}",
        )
        print(f"[upload] done -> https://huggingface.co/{args.repo_id}/tree/main/{path_in_repo}")


if __name__ == "__main__":
    main()
