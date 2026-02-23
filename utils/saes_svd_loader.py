# utils/saes_svd_loader.py

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedLinearAB(nn.Module):
    """Low-rank approximation of nn.Linear using factors A/B.

    Convention (normalized by loader):
      A: (rank, in_features)
      B: (out_features, rank)
      bias: (out_features,) optional

    Weight approximation: W ≈ B @ A

    Forward: y = x @ W^T + b
             = x @ A^T @ B^T + b
    """

    def __init__(self, A: torch.Tensor, B: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError(f"A/B must be 2D tensors, got A{tuple(A.shape)} B{tuple(B.shape)}")

        self.rank, self.in_features = A.shape
        self.out_features, r2 = B.shape
        if r2 != self.rank:
            raise ValueError(f"Rank mismatch: A is {tuple(A.shape)}, B is {tuple(B.shape)}")

        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)

        if bias is None:
            self.bias = None
        else:
            if bias.ndim != 1:
                bias = bias.reshape(-1)
            if bias.numel() != self.out_features:
                raise ValueError(
                    f"Bias mismatch: bias has {tuple(bias.shape)} but out_features={self.out_features}"
                )
            self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_r = F.linear(x, self.A, bias=None)          # (..., rank)
        y = F.linear(x_r, self.B, bias=self.bias)     # (..., out_features)
        return y


@dataclass
class SaesItem:
    name: str
    in_features: int
    out_features: int
    rank: int


def looks_like_saes_svd_checkpoint(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "saes_manifest.json"))
        and os.path.exists(os.path.join(path, "saes_state.pt"))
    )


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_manifest_items(manifest: Dict[str, Any]) -> List[SaesItem]:
    items = manifest.get("items", [])
    out: List[SaesItem] = []
    for it in items:
        out.append(
            SaesItem(
                name=str(it["name"]),
                in_features=int(it["in_features"]),
                out_features=int(it["out_features"]),
                rank=int(it["rank"]),
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


def _maybe_transpose_to(t: torch.Tensor, shape: Tuple[int, int]) -> Optional[torch.Tensor]:
    if tuple(t.shape) == tuple(shape):
        return t
    if t.ndim == 2 and tuple(t.t().shape) == tuple(shape):
        return t.t()
    return None


def _normalize_A_B(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    in_features: int,
    out_features: int,
    rank: int,
    name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize raw A/B tensors into expected shapes.

    We normalize into:
      A_norm: (rank, in_features)
      B_norm: (out_features, rank)

    Try:
      1) A:(rank,in), B:(out,rank)
      2) swapped: A:(out,rank), B:(rank,in)  (if naming reversed)
    Also allows transpose in both cases.
    """

    A1 = _maybe_transpose_to(A, (rank, in_features))
    B1 = _maybe_transpose_to(B, (out_features, rank))
    if A1 is not None and B1 is not None:
        return A1, B1

    A2 = _maybe_transpose_to(A, (out_features, rank))
    B2 = _maybe_transpose_to(B, (rank, in_features))
    if A2 is not None and B2 is not None:
        # swap
        return B2, A2

    raise ValueError(
        f"Cannot normalize factors for {name}. "
        f"Expected A/B shapes compatible with (rank,in)=({rank},{in_features}) and (out,rank)=({out_features},{rank}). "
        f"Got A={tuple(A.shape)} B={tuple(B.shape)}."
    )


def apply_saes_svd_from_state(
    model: nn.Module,
    *,
    manifest: Dict[str, Any],
    saes_state: Dict[str, torch.Tensor],
    strict_shapes: bool = True,
) -> nn.Module:
    """Patch the base model by replacing listed Linear modules with FactorizedLinearAB.

    IMPORTANT: Some SAES checkpoints store <name>.bias as an *empty* tensor (0 elements)
    to indicate "no bias". We treat that as bias=None.
    """
    items = _parse_manifest_items(manifest)
    patched = 0
    empty_bias_count = 0

    for it in items:
        A_key = f"{it.name}.A"
        B_key = f"{it.name}.B"
        bias_key = f"{it.name}.bias"

        if A_key not in saes_state or B_key not in saes_state:
            raise KeyError(f"Missing factors for {it.name}: need {A_key} and {B_key}")

        A = saes_state[A_key]
        B = saes_state[B_key]
        bias = saes_state.get(bias_key, None)

        # Normalize A/B into:
        #   A: (rank, in_features)
        #   B: (out_features, rank)
        A, B = _normalize_A_B(
            A,
            B,
            in_features=it.in_features,
            out_features=it.out_features,
            rank=it.rank,
            name=it.name,
        )

        # Bias handling:
        # - If bias key is absent -> None
        # - If bias tensor exists but is empty -> treat as None
        # - Else enforce shape (out_features,)
        if bias is not None:
            if not torch.is_tensor(bias):
                raise TypeError(f"Bias for {it.name} is not a tensor: {type(bias)}")

            # Some checkpoints store empty bias tensors to represent "no bias"
            if bias.numel() == 0:
                bias = None
                empty_bias_count += 1
            else:
                if bias.ndim != 1:
                    bias = bias.reshape(-1)
                if strict_shapes and bias.numel() != it.out_features:
                    raise ValueError(
                        f"Bias for {it.name} has {bias.numel()} elems, expected {it.out_features}"
                    )

        fac = FactorizedLinearAB(
            A.detach().cpu(),
            B.detach().cpu(),
            bias.detach().cpu() if bias is not None else None,
        )

        _set_submodule(model, it.name, fac)
        patched += 1

    print(f"[SAES-SVD] Patched {patched}/{len(items)} modules using .A/.B/.bias")
    if empty_bias_count:
        print(f"[SAES-SVD] Treated {empty_bias_count} empty bias tensors as bias=None")

    return model



def load_saes_svd_model(
    saes_dir_or_repo: str,
    *,
    base_model: str,
    hf_token: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
):
    """Load base HF model + apply SAES-SVD factors from (dir|repo)."""

    # Resolve local dir (download if needed)
    local_dir = saes_dir_or_repo
    if not os.path.isdir(local_dir):
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise RuntimeError(f"huggingface_hub required to download SAES-SVD repo: {e}")

        kwargs = dict(repo_id=saes_dir_or_repo, revision=revision, cache_dir=cache_dir)
        if hf_token:
            try:
                local_dir = snapshot_download(token=hf_token, **kwargs)
            except TypeError:
                local_dir = snapshot_download(use_auth_token=hf_token, **kwargs)
        else:
            local_dir = snapshot_download(**kwargs)

    manifest_path = os.path.join(local_dir, "saes_manifest.json")
    state_path = os.path.join(local_dir, "saes_state.pt")
    if not os.path.exists(manifest_path) or not os.path.exists(state_path):
        raise FileNotFoundError(f"Expected saes_manifest.json + saes_state.pt under: {local_dir}")

    manifest = _read_json(manifest_path)
    raw_state = torch.load(state_path, map_location="cpu")
    if not isinstance(raw_state, dict):
        raise TypeError(f"saes_state.pt must be a dict/OrderedDict, got {type(raw_state)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    def _from_pretrained(cls, name_or_path: str, **kwargs):
        if hf_token:
            try:
                return cls.from_pretrained(name_or_path, token=hf_token, **kwargs)
            except TypeError:
                return cls.from_pretrained(name_or_path, use_auth_token=hf_token, **kwargs)
        return cls.from_pretrained(name_or_path, **kwargs)

    # Tokenizer: prefer SAES directory
    try:
        tokenizer = _from_pretrained(AutoTokenizer, local_dir, trust_remote_code=trust_remote_code, use_fast=True)
    except Exception:
        tokenizer = _from_pretrained(AutoTokenizer, base_model, trust_remote_code=trust_remote_code, use_fast=True)

    # Base model
    model = _from_pretrained(
        AutoModelForCausalLM,
        base_model,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )

    # Resize embeddings to match SAES embed weight if provided
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

    # Load plain keys that exist in base model state_dict (e.g., layernorms/embeddings)
    base_sd = model.state_dict()
    plain = {k: v for k, v in raw_state.items() if k in base_sd and torch.is_tensor(v)}
    model.load_state_dict(plain, strict=False)

    # Patch factorized modules
    model = apply_saes_svd_from_state(model, manifest=manifest, saes_state=raw_state)

    try:
        model.tie_weights()
    except Exception:
        pass

    meta = {
        "checkpoint_dir": local_dir,
        "base_model": base_model,
        "saes_version": manifest.get("saes_version"),
        "num_manifest_items": len(manifest.get("items", [])) if isinstance(manifest.get("items"), list) else None,
        "num_plain_loaded": len(plain),
    }
    return model, tokenizer, meta
