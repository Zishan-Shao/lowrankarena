# utils/df_svd_loader.py

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class DFSVDFactorizedLinear(nn.Module):
    """DF-SVD factorized linear module.

    Matches the structure saved by df_svd.py:
      buffers:  <name>.Wv, <name>.Wp
      params:   <name>.Bm, <name>.Am   (only if update_rank > 0)

    Row-major forward:
      xk = x @ Wv^T                (..., k)
      Wu = Wp + Bm @ Am            (out, k)
      y  = xk @ Wu^T               (..., out)

    IMPORTANT: Factors are meant to stay in a "safe" dtype (usually float32) even when
    the rest of the model is cast to bf16/fp16. We therefore override _apply to keep
    factor dtypes fixed while still allowing device moves.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank_k: int,
        update_rank: int,
        *,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank_k = int(rank_k)
        self.update_rank = int(update_rank)
        self.factor_dtype = dtype

        self.register_buffer("Wv", torch.empty((self.rank_k, self.in_features), dtype=dtype))
        self.register_buffer("Wp", torch.empty((self.out_features, self.rank_k), dtype=dtype))

        if self.update_rank > 0:
            self.Bm = nn.Parameter(torch.empty((self.out_features, self.update_rank), dtype=dtype))
            self.Am = nn.Parameter(torch.empty((self.update_rank, self.rank_k), dtype=dtype))
        else:
            self.Bm = None
            self.Am = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        y2d = self.forward_flat(x2d)
        return y2d.reshape(*orig_shape[:-1], self.out_features)

    def forward_flat(self, x2d: torch.Tensor) -> torch.Tensor:
        in_dtype = x2d.dtype
        xk = x2d.to(self.Wv.dtype) @ self.Wv.t()
        Wu = self.Wp
        if self.update_rank > 0 and self.Bm is not None and self.Am is not None:
            Wu = Wu + (self.Bm @ self.Am)
        y = xk @ Wu.t()
        if y.dtype != in_dtype:
            # Clamp before casting back to prevent inf/NaN from overflow.
            try:
                if in_dtype.is_floating_point:
                    maxv = torch.finfo(in_dtype).max
                    y = torch.clamp(y, min=-maxv, max=maxv)
            except Exception:
                pass
            y = y.to(in_dtype)
        return y

    # --- dtype-preserving cast behavior ---
    def _apply(self, fn):
        # Let PyTorch move tensors (and maybe cast them)...
        super()._apply(fn)

        # ...then restore factor dtype while keeping the (possibly new) device.
        target_dtype = self.factor_dtype
        try:
            self.Wv.data = self.Wv.data.to(dtype=target_dtype)
            self.Wp.data = self.Wp.data.to(dtype=target_dtype)
            if self.Bm is not None:
                self.Bm.data = self.Bm.data.to(dtype=target_dtype)
            if self.Am is not None:
                self.Am.data = self.Am.data.to(dtype=target_dtype)
        except Exception:
            pass
        return self


@dataclass
class DFSVDItem:
    name: str
    in_features: int
    out_features: int
    rank_k: int
    update_rank: int


def looks_like_dfsvd_checkpoint(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "dfsvd_manifest.json"))
        and os.path.exists(os.path.join(path, "dfsvd_state.pt"))
    )


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_manifest_items(manifest: Dict[str, Any]) -> List[DFSVDItem]:
    items = manifest.get("items", [])
    out: List[DFSVDItem] = []
    for it in items:
        out.append(
            DFSVDItem(
                name=str(it["name"]),
                in_features=int(it["in_features"]),
                out_features=int(it["out_features"]),
                rank_k=int(it["rank_k"]),
                update_rank=int(it["update_rank"]),
            )
        )
    return out


def _get_submodule(root: nn.Module, path: str) -> nn.Module:
    if not path:
        return root
    if hasattr(root, "get_submodule"):
        return root.get_submodule(path)  # type: ignore[attr-defined]
    cur = root
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _set_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    if "." not in path:
        setattr(root, path, module)
        return
    parent_path, leaf = path.rsplit(".", 1)
    parent = _get_submodule(root, parent_path)
    setattr(parent, leaf, module)


def apply_dfsvd_structure(
    model: nn.Module,
    *,
    manifest: Dict[str, Any],
    factor_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Replace the listed Linear modules with DFSVDFactorizedLinear modules.

    This only patches the *structure* (module replacement). You still need to call
    model.load_state_dict(dfsvd_state, strict=False) afterwards.
    """

    items = _parse_manifest_items(manifest)
    patched = 0
    for it in items:
        fac = DFSVDFactorizedLinear(
            in_features=it.in_features,
            out_features=it.out_features,
            rank_k=it.rank_k,
            update_rank=it.update_rank,
            dtype=factor_dtype,
        )
        _set_submodule(model, it.name, fac)
        patched += 1
    print(f"[DF-SVD] Patched {patched}/{len(items)} modules using .Wv/.Wp/.Bm/.Am")
    return model


def load_dfsvd_model(
    dfsvd_dir_or_repo: str,
    *,
    base_model: str,
    hf_token: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
    factor_dtype: torch.dtype = torch.float32,
):
    """Load base HF model + apply DF-SVD factors from (dir|repo).

    DF-SVD checkpoints are expected to contain:
      - dfsvd_manifest.json
      - dfsvd_state.pt
      - (optional) tokenizer files

    Because DF-SVD dir does not contain a full HF config, you must supply base_model.
    """

    # Resolve local dir (download if needed)
    local_dir = dfsvd_dir_or_repo
    if not os.path.isdir(local_dir):
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise RuntimeError(f"huggingface_hub required to download DF-SVD repo: {e}")

        kwargs = dict(repo_id=dfsvd_dir_or_repo, revision=revision, cache_dir=cache_dir)
        if hf_token:
            try:
                local_dir = snapshot_download(token=hf_token, **kwargs)
            except TypeError:
                local_dir = snapshot_download(use_auth_token=hf_token, **kwargs)
        else:
            local_dir = snapshot_download(**kwargs)

    manifest_path = os.path.join(local_dir, "dfsvd_manifest.json")
    state_path = os.path.join(local_dir, "dfsvd_state.pt")
    if not os.path.exists(manifest_path) or not os.path.exists(state_path):
        raise FileNotFoundError(f"Expected dfsvd_manifest.json + dfsvd_state.pt under: {local_dir}")

    manifest = _read_json(manifest_path)
    raw_state = torch.load(state_path, map_location="cpu")
    if not isinstance(raw_state, dict):
        raise TypeError(f"dfsvd_state.pt must be a dict/OrderedDict, got {type(raw_state)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    def _from_pretrained(cls, name_or_path: str, **kwargs):
        if hf_token:
            try:
                return cls.from_pretrained(name_or_path, token=hf_token, **kwargs)
            except TypeError:
                return cls.from_pretrained(name_or_path, use_auth_token=hf_token, **kwargs)
        return cls.from_pretrained(name_or_path, **kwargs)

    # Tokenizer: prefer DF-SVD directory
    try:
        tokenizer = _from_pretrained(
            AutoTokenizer, local_dir, trust_remote_code=trust_remote_code, use_fast=True
        )
    except Exception:
        tokenizer = _from_pretrained(
            AutoTokenizer, base_model, trust_remote_code=trust_remote_code, use_fast=True
        )

    # Base model
    model = _from_pretrained(
        AutoModelForCausalLM,
        base_model,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )

    # Resize embeddings if needed (match checkpoint vocab if present)
    try:
        if "model.embed_tokens.weight" in raw_state and torch.is_tensor(raw_state["model.embed_tokens.weight"]):
            target_vocab = int(raw_state["model.embed_tokens.weight"].shape[0])
        else:
            target_vocab = len(tokenizer)
        cur_vocab = int(model.get_input_embeddings().weight.shape[0])
        if target_vocab and cur_vocab != target_vocab:
            model.resize_token_embeddings(target_vocab)
    except Exception:
        pass

    # Patch factorized modules first, so state_dict can load DF-SVD keys
    model = apply_dfsvd_structure(model, manifest=manifest, factor_dtype=factor_dtype)

    # Load all weights (plain + DF-SVD factors)
    missing, unexpected = model.load_state_dict(raw_state, strict=False)
    if missing:
        print(f"[DF-SVD] Missing keys (showing first 20): {missing[:20]}")
    if unexpected:
        print(f"[DF-SVD] Unexpected keys (showing first 20): {unexpected[:20]}")

    try:
        model.tie_weights()
    except Exception:
        pass

    meta = {
        "checkpoint_dir": local_dir,
        "base_model": base_model,
        "dfsvd_version": manifest.get("dfsvd_version"),
        "num_manifest_items": len(manifest.get("items", [])) if isinstance(manifest.get("items"), list) else None,
    }
    return model, tokenizer, meta
