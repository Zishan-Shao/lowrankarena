from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Type

import torch
import torch.nn as nn


def find_layers(
    module: nn.Module,
    layers: Tuple[Type[nn.Module], ...] = (nn.Linear,),
    *,
    prefix: str = "",
) -> Dict[str, nn.Module]:
    """Recursively collect layers of given types.

    The returned keys match `module.named_modules()` names, which is what the
    SVD-LLM scripts expect (e.g. "self_attn.q_proj", "mlp.gate_proj").
    """
    found: Dict[str, nn.Module] = {}
    for name, child in module.named_modules():
        if name == "":
            continue
        if isinstance(child, layers):
            found[prefix + name] = child
    return found


def _token_arg(hf_token: Optional[str]):
    # transformers>=4.35 uses `token=`; older versions use `use_auth_token=`.
    return {"token": hf_token} if hf_token else {}


def _is_tokenizer_like(obj) -> bool:
    if obj is None or isinstance(obj, bool):
        return False
    # HF tokenizers are callable and implement encode/decode.
    if callable(obj):
        return True
    return hasattr(obj, "encode") and hasattr(obj, "decode")


def _load_tokenizer(model_id: str, *, hf_token: Optional[str] = None):
    from transformers import AutoTokenizer

    last_err: Optional[Exception] = None
    for use_fast in (True, False):
        try:
            tok = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=use_fast,
                **_token_arg(hf_token),
            )
        except Exception as e:
            last_err = e
            continue

        if not _is_tokenizer_like(tok):
            # Extremely defensive: we've observed environments where a non-tokenizer
            # placeholder (e.g. bool) can surface here.
            continue

        try:
            if getattr(tok, "pad_token", None) is None:
                tok.pad_token = tok.eos_token
        except Exception:
            pass
        return tok

    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Failed to load a usable tokenizer for: {model_id}")


def ensure_transformers_layer_idx(model: nn.Module) -> None:
    """Ensure decoder self-attention modules have `layer_idx`.

    Transformers>=4.4x Cache implementations (DynamicCache/StaticCache/...) use
    `layer_idx` to route KV updates. Some pickled checkpoints (e.g. after module
    replacement) may miss this attribute.
    """
    try:
        # LLaMA / Mistral style
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is not None:
            for i, layer in enumerate(layers):
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    # Force-correct even if present: some replacement pipelines set all to 0.
                    setattr(attn, "layer_idx", int(i))
            return
        # OPT style
        dec_layers = getattr(getattr(getattr(model, "model", None), "decoder", None), "layers", None)
        if dec_layers is not None:
            for i, layer in enumerate(dec_layers):
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    setattr(attn, "layer_idx", int(i))
    except Exception:
        # Best-effort only.
        return


def _ensure_inline_rotary_emb(llama_model: nn.Module, config) -> None:
    """Attach a minimal rotary-embedding module when the HF class cannot be instantiated.

    Returns (cos, sin) tensors of shape [B, 1, S, head_dim] to match the
    position_embeddings contract used by transformers>=4.43 LlamaAttention.
    """
    head_dim = getattr(config, "head_dim",
                       config.hidden_size // config.num_attention_heads)
    rope_theta = float(getattr(config, "rope_theta", 10000.0))
    max_pos = int(getattr(config, "max_position_embeddings", 4096))

    class _InlineRoPE(nn.Module):
        def __init__(self):
            super().__init__()
            inv = 1.0 / (rope_theta ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
            ))
            self.register_buffer("inv_freq", inv, persistent=False)
            self.rope_type = "default"  # sentinel so compat check passes

        def forward(self, x, position_ids=None, seq_len=None):
            device = x.device
            if position_ids is None:
                s = x.shape[1] if x.dim() >= 2 else (seq_len or max_pos)
                position_ids = torch.arange(s, device=device).unsqueeze(0)
            inv = self.inv_freq.to(device=device, dtype=torch.float32)
            # [B, S, head_dim//2]
            freqs = torch.einsum("bi,j->bij", position_ids.float(), inv)
            emb = torch.cat([freqs, freqs], dim=-1)          # [B, S, head_dim]
            cos = emb.cos()                                  # [B, S, head_dim]
            sin = emb.sin()
            return cos.to(x.dtype), sin.to(x.dtype)

    llama_model.rotary_emb = _InlineRoPE()
    print("[compat] Attached inline RoPE fallback to LlamaModel")


def ensure_model_level_rotary_emb(model: nn.Module) -> None:
    """Patch old pickled LlamaForCausalLM for transformers>=4.43/4.48 compatibility.

    Changes between old pickled models and transformers 4.48:
      - LlamaModel: rotary_emb moved from each LlamaAttention to LlamaModel level
      - LlamaRotaryEmbedding: constructor changed (dim,max_pos,base) → (config,); added rope_type
      - LlamaAttention: added scaling, attention_dropout, is_causal; uses passed position_embeddings
    """
    try:
        llama_model = getattr(model, "model", None)
        if llama_model is None:
            return
        config = model.config

        # ── 1. Model-level rotary_emb ───────────────────────────────────────────
        existing_rope = getattr(llama_model, "rotary_emb", None)
        if existing_rope is None or not hasattr(existing_rope, "rope_type"):
            try:
                from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
                # transformers>=4.50: LlamaRotaryEmbedding reads config.rope_parameters
                # which is not present on old pickled configs.  Pre-populate it via
                # ROPE_INIT_FUNCTIONS so the constructor succeeds.
                if not hasattr(config, "rope_parameters"):
                    try:
                        from transformers.models.llama.modeling_llama import ROPE_INIT_FUNCTIONS
                        rope_type = "default"
                        if hasattr(config, "rope_scaling") and config.rope_scaling:
                            rope_type = (
                                config.rope_scaling.get("rope_type")
                                or config.rope_scaling.get("type")
                                or "default"
                            )
                        inv_freq, attn_factor = ROPE_INIT_FUNCTIONS[rope_type](config, device=None)
                        config.rope_parameters = (inv_freq, attn_factor)
                    except Exception:
                        pass
                llama_model.rotary_emb = LlamaRotaryEmbedding(config=config)
                print("[compat] Created fresh model-level LlamaRotaryEmbedding from config")
            except Exception as e:
                print(f"[compat] Could not create LlamaRotaryEmbedding: {e}")
                # Last-resort: inline implementation that produces (cos, sin) matching the
                # shape expected by LlamaAttention.forward in transformers 4.43+.
                _ensure_inline_rotary_emb(llama_model, config)

        # ── 2. Per-attention-layer missing attributes ───────────────────────────
        import math
        layers = getattr(llama_model, "layers", None) or []
        patched_attn = 0
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            head_dim = getattr(attn, "head_dim",
                               getattr(config, "head_dim",
                                       config.hidden_size // config.num_attention_heads))
            if not hasattr(attn, "scaling"):
                attn.scaling = head_dim ** -0.5
                patched_attn += 1
            if not hasattr(attn, "attention_dropout"):
                attn.attention_dropout = getattr(config, "attention_dropout", 0.0)
            if not hasattr(attn, "is_causal"):
                attn.is_causal = True
            if not hasattr(attn, "config"):
                attn.config = config
        if patched_attn:
            print(f"[compat] Patched {patched_attn} LlamaAttention layers with missing 4.48 attributes")
    except Exception as e:
        print(f"[compat] ensure_model_level_rotary_emb failed: {e}")


def get_model_from_huggingface(
    model_id: str,
    *,
    hf_token: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    """Load (model, tokenizer) from HuggingFace hub.

    Note: we keep the model on CPU; callers can move/cast as needed.
    """
    from transformers import AutoModelForCausalLM

    tok = _load_tokenizer(model_id, hf_token=hf_token)

    # Avoid HF warnings about dtype; let caller cast later if desired.
    kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True, **_token_arg(hf_token))
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    ensure_transformers_layer_idx(model)

    # Common in quant/ptq pipelines.
    model.seqlen = int(getattr(model.config, "max_position_embeddings", 2048))
    return model, tok


def get_model_from_local(path: str):
    """Load (model, tokenizer) from a local torch.save checkpoint.

    Expected format: torch.save({'model': model, 'tokenizer': tokenizer}, path)
    """
    p = Path(path)
    if p.is_dir():
        from transformers import AutoModelForCausalLM

        tok = _load_tokenizer(str(p), hf_token=None)
        model = AutoModelForCausalLM.from_pretrained(str(p), trust_remote_code=True, low_cpu_mem_usage=True)
        ensure_transformers_layer_idx(model)
        model.seqlen = int(getattr(model.config, "max_position_embeddings", 2048))
        return model, tok

    obj = _torch_load_local_checkpoint(p)
    if isinstance(obj, dict) and "model" in obj and "tokenizer" in obj:
        model = obj["model"]
        ensure_transformers_layer_idx(model)
        ensure_model_level_rotary_emb(model)
        tok = obj["tokenizer"]
        # Pickled tokenizers can be incomplete (missing _special_tokens_map etc.)
        # after module-path shims. Validate and reload from model_id if broken.
        if not _is_tokenizer_like(tok) or not hasattr(tok, "_special_tokens_map"):
            model_id = getattr(getattr(model, "config", None), "_name_or_path", None)
            if model_id:
                try:
                    tok = _load_tokenizer(str(model_id), hf_token=None)
                    print(f"[warn] Pickled tokenizer was broken; reloaded from: {model_id}")
                except Exception as e:
                    print(f"[warn] Could not reload tokenizer from {model_id}: {e}")
        return model, tok
    if hasattr(obj, "forward"):
        # A pickled HF model (rare but possible): try to recover tokenizer.
        model = obj
        ensure_transformers_layer_idx(model)
        ensure_model_level_rotary_emb(model)
        model_id = getattr(model, "name_or_path", None)
        if model_id:
            try:
                tok = _load_tokenizer(str(model_id), hf_token=None)
                return model, tok
            except Exception:
                pass
        raise ValueError(
            f"Loaded a model object from {path} but could not recover a tokenizer. "
            "Please re-save as {'model': model, 'tokenizer': tok}."
        )
    raise ValueError(f"Unrecognized checkpoint format at: {path}")


@torch.no_grad()
def measure_param_bytes(model: nn.Module) -> int:
    """Sum unique parameter storage bytes (avoids double-count on tied weights)."""
    seen = set()
    total = 0
    for p in model.parameters():
        if p is None:
            continue
        try:
            st = p.untyped_storage()
            key = (int(st.data_ptr()), int(st.nbytes()))
            nbytes = int(st.nbytes())
        except Exception:
            st = p.storage()
            nbytes = int(st.size()) * int(p.element_size())
            key = (int(st.data_ptr()), int(nbytes))
        if key in seen:
            continue
        seen.add(key)
        total += nbytes
    return total


def mib(nbytes: int) -> float:
    return float(nbytes) / (1024.0**2)


def set_env_flag(name: str, value: bool):
    if value:
        os.environ[name] = "1"
    else:
        # Don't leave stale "0" around; scripts typically check != "0".
        os.environ.pop(name, None)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _looks_like_flashsvd_checkpoint(path: Path) -> bool:
    try:
        root = Path(__file__).resolve().parents[1]  # FlashSVD-v1.5/
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            return False
    except Exception:
        # Best-effort: if we can't resolve, fall back to name heuristics only.
        pass
    name = path.name.lower()
    # Common outputs produced by SVDLLM(.py) / SVDLLM_flashsvd(.py)
    return any(
        key in name
        for key in (
            "_whitening_only_",
            "_whitening_then_update_",
            "_update_only_",
            "_profiling_",
            "svdllm",
            "flashsvd",
            "merge.pt",
        )
    )


def _inject_pickle_shims():
    """Inject sys.modules shims for transformers modules that moved between versions.

    Checkpoints pickled with older transformers reference full module paths
    (e.g. transformers.models.llama.tokenization_llama_fast).  If those paths
    no longer exist in the installed version, unpickling raises ModuleNotFoundError.
    We redirect each old path to wherever the class now lives.
    """
    import sys
    import types

    _SHIM_MAP = {
        # transformers >=4.44 removed the per-model tokenizer_fast sub-modules;
        # classes moved to transformers.models.llama (re-exported from the package).
        "transformers.models.llama.tokenization_llama_fast": (
            "transformers.models.llama",
            ["LlamaTokenizerFast"],
        ),
        "transformers.models.llama.tokenization_llama": (
            "transformers.models.llama",
            ["LlamaTokenizer"],
        ),
        "transformers.models.mistral.tokenization_mistral_fast": (
            "transformers.models.mistral",
            ["MistralTokenizerFast"],
        ),
    }

    for old_path, (new_mod_path, attrs) in _SHIM_MAP.items():
        if old_path in sys.modules:
            continue
        try:
            new_mod = __import__(new_mod_path, fromlist=attrs)
        except ImportError:
            continue
        shim = types.ModuleType(old_path)
        for attr in attrs:
            obj = getattr(new_mod, attr, None)
            if obj is not None:
                setattr(shim, attr, obj)
        sys.modules[old_path] = shim


def _torch_load_local_checkpoint(path: Path):
    """Load a local torch checkpoint, handling PyTorch>=2.6 weights_only default.

    PyTorch 2.6 changed `torch.load` default `weights_only=True`, which rejects
    pickled model/tokenizer objects (our SVDLLM scripts save them as a dict).
    We try a safe load first and fall back to `weights_only=False` only when we
    can reasonably assume the checkpoint is trusted.
    """

    def _load(*, weights_only: Optional[bool]):
        if weights_only is None:
            return torch.load(str(path), map_location="cpu")
        try:
            return torch.load(str(path), map_location="cpu", weights_only=weights_only)
        except TypeError:
            # Older PyTorch without weights_only.
            return torch.load(str(path), map_location="cpu")

    try:
        return _load(weights_only=True)
    except Exception as e:
        msg = str(e)
        is_weights_only = "Weights only load failed" in msg or "weights_only" in msg
        if not is_weights_only:
            raise

        trusted = _env_truthy("FLASH_SVD_TRUST_PICKLE") or _looks_like_flashsvd_checkpoint(path)
        if not trusted:
            raise RuntimeError(
                f"Refusing to load pickled checkpoint with weights_only=False: {path}\n"
                "PyTorch>=2.6 defaults torch.load(weights_only=True), which can't load "
                "our pickled {'model': model, 'tokenizer': tok} format.\n"
                "If you trust this checkpoint, re-run with: FLASH_SVD_TRUST_PICKLE=1"
            ) from e

        # Trusted fallback: allow pickled model/tokenizer objects.
        # Older checkpoints were pickled with transformers modules that may have moved
        # in newer versions.  Inject sys.modules shims so unpickling still works.
        _inject_pickle_shims()
        return _load(weights_only=False)
