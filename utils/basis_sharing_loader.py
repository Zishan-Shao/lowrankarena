from __future__ import annotations

import importlib
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dedupe_keep_order(items: Iterable[Optional[str]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _looks_like_basis_sharing_config(cfg: Dict[str, Any]) -> bool:
    architectures = cfg.get("architectures", [])
    if isinstance(architectures, list) and any(str(x).startswith("Share") for x in architectures):
        return True

    basis_keys = [k for k in cfg.keys() if str(k).startswith("num_basis_")]
    group_keys = [k for k in cfg.keys() if str(k).endswith("_groups")]
    return bool(basis_keys) and bool(group_keys)


def looks_like_basis_sharing_checkpoint(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    cfg_path = os.path.join(path, "config.json")
    if not os.path.exists(cfg_path):
        return False
    try:
        cfg = _read_json(cfg_path)
    except Exception:
        return False
    return _looks_like_basis_sharing_config(cfg)


def _resolve_local_dir(
    basis_dir_or_repo: str,
    *,
    hf_token: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    if os.path.isdir(basis_dir_or_repo):
        return basis_dir_or_repo

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(f"huggingface_hub required to download Basis Sharing repo: {e}")

    kwargs = dict(repo_id=basis_dir_or_repo, revision=revision, cache_dir=cache_dir)
    if hf_token:
        try:
            return snapshot_download(token=hf_token, **kwargs)
        except TypeError:
            return snapshot_download(use_auth_token=hf_token, **kwargs)
    return snapshot_download(**kwargs)


def _candidate_basis_code_dirs(user_dir: Optional[str] = None) -> List[str]:
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = _dedupe_keep_order(
        [
            user_dir,
            os.environ.get("BASIS_SHARING_DIR"),
            os.path.abspath(os.path.join(here, "..", "baselines", "Basis_Sharing")),
            os.path.abspath(os.path.join(os.getcwd(), "baselines", "Basis_Sharing")),
        ]
    )
    return [p for p in candidates if os.path.isdir(p)]


def _ensure_basis_code_on_syspath(basis_code_dir: str) -> None:
    models_dir = os.path.join(basis_code_dir, "models")
    for p in (basis_code_dir, models_dir):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def _import_basis_model_cls(model_type: str, basis_code_dir: Optional[str] = None):
    candidates = _candidate_basis_code_dirs(basis_code_dir)
    if not candidates:
        raise FileNotFoundError(
            "Could not find the vendored Basis_Sharing code directory. "
            "Set basis_code_dir=..., or set BASIS_SHARING_DIR, or place the repo at baselines/Basis_Sharing."
        )

    last_err: Optional[Exception] = None
    for code_dir in candidates:
        try:
            _ensure_basis_code_on_syspath(code_dir)
            mt = (model_type or "").lower()
            if mt in {"llama", "llama2"}:
                module = importlib.import_module("models.llama")
                return getattr(module, "ShareLlamaForCausalLM"), code_dir
            if mt == "gpt2":
                module = importlib.import_module("models.gpt2")
                return getattr(module, "ShareGPT2LMHeadModel"), code_dir
            if mt == "opt":
                module = importlib.import_module("models.opt")
                return getattr(module, "ShareOPTForCausalLM"), code_dir
            if mt == "mistral":
                module = importlib.import_module("models.mistral")
                return getattr(module, "ShareMistralForCausalLM"), code_dir
            raise ValueError(f"Unsupported Basis Sharing model_type: {model_type}")
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        "Failed to import the Basis Sharing custom model class. "
        "This usually means the vendored Basis_Sharing code is missing, not on sys.path, "
        "or still needs the Transformers-compatibility patch."
    ) from last_err


def _from_pretrained(cls, name_or_path: str, *, hf_token: Optional[str] = None, **kwargs):
    if hf_token:
        try:
            return cls.from_pretrained(name_or_path, token=hf_token, **kwargs)
        except TypeError:
            return cls.from_pretrained(name_or_path, use_auth_token=hf_token, **kwargs)
    return cls.from_pretrained(name_or_path, **kwargs)


def _pick_tokenizer_source(
    local_dir: str,
    *,
    tokenizer_name: Optional[str],
    base_model: Optional[str],
    config: Any,
) -> List[str]:
    # Prefer an explicitly provided tokenizer source, but still try the checkpoint first.
    cfg_name_or_path = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)
    return _dedupe_keep_order([local_dir, tokenizer_name, base_model, cfg_name_or_path])


def _load_tokenizer(
    *,
    local_dir: str,
    tokenizer_name: Optional[str],
    base_model: Optional[str],
    config: Any,
    hf_token: Optional[str],
    trust_remote_code: bool,
    use_fast: Optional[bool],
):
    from transformers import AutoTokenizer

    last_err: Optional[Exception] = None
    for src in _pick_tokenizer_source(local_dir, tokenizer_name=tokenizer_name, base_model=base_model, config=config):
        try:
            kwargs = dict(trust_remote_code=trust_remote_code)
            if use_fast is not None:
                kwargs["use_fast"] = use_fast
            tok = _from_pretrained(AutoTokenizer, src, hf_token=hf_token, **kwargs)
            return tok, src
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        "Failed to load a tokenizer for the Basis Sharing checkpoint. "
        "If this is a trained checkpoint directory that only has model weights/config, pass tokenizer_name=... or base_model=... ."
    ) from last_err


def load_basis_sharing_model(
    basis_dir_or_repo: str,
    *,
    base_model: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    basis_code_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[Any] = "auto",
    low_cpu_mem_usage: bool = True,
    attn_implementation: Optional[str] = None,
    use_fast_tokenizer: Optional[bool] = False,
):
    """Load a Basis Sharing checkpoint using the vendored custom model code.

    This is analogous to the SAES-SVD loader: it resolves a local directory or HF repo,
    loads the tokenizer (preferring the checkpoint dir, then falling back to tokenizer_name/base_model),
    imports the correct Basis Sharing model class from the vendored code, and returns
    (model, tokenizer, meta).

    Parameters
    ----------
    basis_dir_or_repo:
        Local checkpoint directory or Hugging Face repo id.
    base_model:
        Optional fallback tokenizer source. Useful when the trained checkpoint dir does
        not include tokenizer files.
    tokenizer_name:
        Optional explicit tokenizer source; checked before base_model fallback.
    basis_code_dir:
        Path to the vendored Basis_Sharing repo root. If omitted, the loader tries
        BASIS_SHARING_DIR, ../baselines/Basis_Sharing, and ./baselines/Basis_Sharing.
    """
    from transformers import AutoConfig

    local_dir = _resolve_local_dir(
        basis_dir_or_repo,
        hf_token=hf_token,
        revision=revision,
        cache_dir=cache_dir,
    )

    cfg_path = os.path.join(local_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Expected config.json under Basis Sharing checkpoint: {local_dir}")

    raw_cfg = _read_json(cfg_path)
    if not _looks_like_basis_sharing_config(raw_cfg):
        raise ValueError(
            f"{local_dir} does not look like a Basis Sharing checkpoint "
            f"(missing Basis Sharing-specific config keys like num_basis_* / *_groups)."
        )

    config = _from_pretrained(AutoConfig, local_dir, hf_token=hf_token, trust_remote_code=trust_remote_code)
    model_type = getattr(config, "model_type", None)

    ModelCls, resolved_code_dir = _import_basis_model_cls(model_type, basis_code_dir=basis_code_dir)

    tokenizer, tokenizer_source = _load_tokenizer(
        local_dir=local_dir,
        tokenizer_name=tokenizer_name,
        base_model=base_model,
        config=config,
        hf_token=hf_token,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast_tokenizer,
    )

    model_kwargs: Dict[str, Any] = dict(trust_remote_code=trust_remote_code)
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if low_cpu_mem_usage:
        model_kwargs["low_cpu_mem_usage"] = True
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = _from_pretrained(ModelCls, local_dir, hf_token=hf_token, **model_kwargs)

    # Friendly defaults for eval.
    try:
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        pass
    try:
        if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass
    try:
        model.seqlen = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", None)
    except Exception:
        pass
    try:
        model.tie_weights()
    except Exception:
        pass

    meta = {
        "checkpoint_dir": local_dir,
        "basis_code_dir": resolved_code_dir,
        "model_type": model_type,
        "architectures": raw_cfg.get("architectures"),
        "tokenizer_source": tokenizer_source,
        "base_model_fallback": base_model,
        "num_basis": {k: raw_cfg[k] for k in raw_cfg if str(k).startswith("num_basis_")},
        "group_keys": sorted([k for k in raw_cfg if str(k).endswith("_groups")]),
    }
    return model, tokenizer, meta
