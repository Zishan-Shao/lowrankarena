#coding:utf8
import os
import sys
import torch
import torch.nn as nn

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

# bandaid fix
dev = torch.device("cuda")


def _resolve_hf_token(hf_token: str = None):
    if hf_token is not None:
        return hf_token
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )


def _tokenizer_ok(tokenizer) -> bool:
    try:
        if tokenizer is None or isinstance(tokenizer, bool):
            return False
        if callable(tokenizer):
            return True
        # Be tolerant of older/newer pickled tokenizer objects that may not be
        # directly callable but still expose the standard tokenizer API.
        has_min_api = (
            hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode")
        ) or (
            hasattr(tokenizer, "pad_token_id") and hasattr(tokenizer, "eos_token_id")
        )
        return bool(has_min_api)
    except Exception:
        return False


def _ensure_tokenizer_pad(tokenizer):
    if tokenizer is None:
        return tokenizer
    try:
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        pass
    return tokenizer


def _load_tokenizer_from_hint(model_hint: str, hf_token: str = None):
    if not model_hint:
        return None
    hf_token = _resolve_hf_token(hf_token)
    model_hint = str(model_hint)

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            model_hint, trust_remote_code=True, use_fast=True, token=hf_token
        )
        if _tokenizer_ok(tok):
            return _ensure_tokenizer_pad(tok)
    except Exception:
        pass
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            model_hint, trust_remote_code=True, use_fast=False, token=hf_token
        )
        if _tokenizer_ok(tok):
            return _ensure_tokenizer_pad(tok)
    except Exception:
        pass

    # Some llama-family repos resolve more reliably through explicit classes.
    try:
        from transformers import LlamaTokenizerFast, LlamaTokenizer
        for cls in (LlamaTokenizerFast, LlamaTokenizer):
            try:
                tok = cls.from_pretrained(model_hint, token=hf_token)
                if _tokenizer_ok(tok):
                    return _ensure_tokenizer_pad(tok)
            except Exception:
                continue
    except Exception:
        pass

    if "llama" in model_hint.lower() or "vicuna" in model_hint.lower():
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                "openlm-research/open_llama_7b", trust_remote_code=True, use_fast=False, token=hf_token
            )
            if _tokenizer_ok(tok):
                return _ensure_tokenizer_pad(tok)
        except Exception:
            pass
    return None


def _checkpoint_path_model_hints(model_path: str):
    if not model_path:
        return []
    model_path = str(model_path)
    lower = model_path.lower()
    hints = []

    explicit_map = {
        "jeffwan_llama_7b_hf": "jeffwan/llama-7b-hf",
        "jeffwan_llama_13b_hf": "jeffwan/llama-13b-hf",
        "jeffwan_llama_30b_hf": "jeffwan/llama-30b-hf",
        "openlm_research_open_llama_7b": "openlm-research/open_llama_7b",
        "openlm_research_open_llama_3b": "openlm-research/open_llama_3b",
    }
    for needle, hint in explicit_map.items():
        if needle in lower and hint not in hints:
            hints.append(hint)

    # Try a conservative owner/repo reconstruction for filenames like:
    #   owner_repo_name_hf_*  -> owner/repo-name-hf
    import os as _os
    import re as _re
    stem = _os.path.basename(lower)
    stem = _re.sub(r"\.pt$", "", stem)
    stem = _re.sub(r"_(whitening|update|svdllm|profiling).*$", "", stem)
    parts = [p for p in stem.split("_") if p]
    if len(parts) >= 3:
        owner = parts[0]
        repo_bits = parts[1:]
        repo = "-".join(repo_bits)
        guess = f"{owner}/{repo}"
        if guess not in hints:
            hints.append(guess)

    return hints


def _recover_tokenizer_from_model(model, tokenizer=None, explicit_hint: str = None, hf_token: str = None, checkpoint_path: str = None):
    hf_token = _resolve_hf_token(hf_token)
    override = (os.getenv("SVDLLM_TOKENIZER_MODEL") or "").strip()

    hints = []
    for hint in (
        override,
        explicit_hint,
        getattr(model, "name_or_path", None),
        getattr(getattr(model, "config", None), "_name_or_path", None),
        getattr(getattr(model, "config", None), "name_or_path", None),
        *_checkpoint_path_model_hints(checkpoint_path),
    ):
        if hint and hint not in hints:
            hints.append(hint)

    # Explicit override takes precedence even if the checkpoint already contains a tokenizer.
    if override:
        tok = _load_tokenizer_from_hint(override, hf_token=hf_token)
        if _tokenizer_ok(tok):
            tokenizer = tok

    if not _tokenizer_ok(tokenizer):
        for hint in hints:
            tok = _load_tokenizer_from_hint(hint, hf_token=hf_token)
            if _tokenizer_ok(tok):
                tokenizer = tok
                break

    # Best-effort vocab sanity check.
    try:
        model_vocab = int(model.get_input_embeddings().weight.shape[0])
    except Exception:
        model_vocab = getattr(getattr(model, "config", None), "vocab_size", None)
        try:
            model_vocab = int(model_vocab) if model_vocab is not None else None
        except Exception:
            model_vocab = None
    tok_vocab = getattr(tokenizer, "vocab_size", None)
    try:
        tok_vocab = int(tok_vocab) if tok_vocab is not None else None
    except Exception:
        tok_vocab = None

    if model_vocab and tok_vocab and model_vocab != tok_vocab:
        for hint in hints:
            tok = _load_tokenizer_from_hint(hint, hf_token=hf_token)
            if _tokenizer_ok(tok):
                new_vocab = getattr(tok, "vocab_size", None)
                try:
                    new_vocab = int(new_vocab) if new_vocab is not None else None
                except Exception:
                    new_vocab = None
                tokenizer = tok
                tok_vocab = new_vocab
                if new_vocab == model_vocab:
                    break

    if not _tokenizer_ok(tokenizer):
        raise TypeError(
            "Tokenizer object is not callable and could not be reconstructed; "
            "set SVDLLM_TOKENIZER_MODEL to a valid local tokenizer path or base model id. "
            f"Tried hints: {hints}"
        )

    return _ensure_tokenizer_pad(tokenizer)


def get_model_from_huggingface(model_id, hf_token: str = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Pick up token from env if not provided
    hf_token = _resolve_hf_token(hf_token)
    # Tokenizer: prefer fast; if protobuf/sentencepiece conversion fails, fall back to slow
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=True, token=hf_token
        )
    except Exception as e:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, use_fast=False, token=hf_token
            )
        except Exception as e2:
            # Some community LLaMA repos lack tokenizer files; fall back to an open LLaMA tokenizer
            if "llama" in model_id.lower() or "vicuna" in model_id.lower():
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        "openlm-research/open_llama_7b", trust_remote_code=True, use_fast=False, token=hf_token
                    )
                    # Ensure pad token exists for batching
                    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                        tokenizer.pad_token = tokenizer.eos_token
                except Exception:
                    raise e2
            else:
                raise e2

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        cache_dir=None,
        token=hf_token,
    )
    # Sequence length hint for downstream code
    model.seqlen = getattr(model.config, 'max_position_embeddings', 2048)
    return model, tokenizer


def get_model_from_local(model_id, hf_token: str = None):
    """
    Load a locally saved checkpoint produced by this repo.

    Supported formats:
      1) Legacy dict checkpoints:
           torch.save({'model': model, 'tokenizer': tokenizer}, path)
      2) Bare model checkpoints (used by newer SVDLLM_v2 scripts):
           torch.save(model, path)

    Older checkpoints were saved with full Python objects. Unpickling those
    requires the same transformers symbols to exist at load time. Some versions
    no longer expose transformers.activations.SiLUActivation, which raises an
    AttributeError while loading.

    To maximize compatibility without forcing a specific transformers version,
    we provide lightweight shims for missing activations/wrappers before
    torch.load.
    """
    hf_token = _resolve_hf_token(hf_token)

    # Provide missing activation and wrapper classes if the installed environment lacks them
    try:
        import transformers
        from transformers import activations as _hf_acts
        if not hasattr(_hf_acts, "SiLUActivation"):
            class SiLUActivation(torch.nn.Module):
                def forward(self, x):  # pragma: no cover - simple shim
                    return torch.nn.functional.silu(x)
            # Attach to the same module path the pickle expects
            setattr(_hf_acts, "SiLUActivation", SiLUActivation)
        # Some checkpoints were saved while wrapper classes lived under __main__
        # (e.g., when using the medium training scripts directly). Expose shims
        # on the current __main__ to make unpickling robust.
        import sys as _sys
        _main = _sys.modules.get("__main__")
        if _main is not None:
            if not hasattr(_main, "WeightLoRAWrapper"):
                class WeightLoRAWrapper(torch.nn.Module):
                    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float, freeze_base: bool = True):
                        super().__init__()
                        self.base = base
                        if freeze_base:
                            for p in self.base.parameters():
                                p.requires_grad = False
                        self.rank = max(int(rank), 1)
                        self.scaling = float(alpha) / float(self.rank)
                        # Lazily created if missing in older checkpoints
                        if not hasattr(self, "lora_down") or not isinstance(getattr(self, "lora_down"), torch.nn.Linear):
                            self.lora_down = torch.nn.Linear(self.base.in_features, self.rank, bias=False,
                                                             device=self.base.weight.device, dtype=self.base.weight.dtype)
                            torch.nn.init.normal_(self.lora_down.weight, mean=0.0, std=0.02)
                        if not hasattr(self, "lora_up") or not isinstance(getattr(self, "lora_up"), torch.nn.Linear):
                            self.lora_up = torch.nn.Linear(self.rank, self.base.out_features, bias=False,
                                                           device=self.base.weight.device, dtype=self.base.weight.dtype)
                            torch.nn.init.normal_(self.lora_up.weight, mean=0.0, std=0.02)
                    def forward(self, x):
                        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scaling
                setattr(_main, "WeightLoRAWrapper", WeightLoRAWrapper)
            if not hasattr(_main, "ActivationSpaceLoRAWrapper"):
                class ActivationSpaceLoRAWrapper(torch.nn.Module):
                    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float, freeze_base: bool = True):
                        super().__init__()
                        self.base = base
                        if freeze_base:
                            for p in self.base.parameters():
                                p.requires_grad = False
                        self.rank = max(int(rank), 1)
                        self.scaling = float(alpha) / float(self.rank)
                        if not hasattr(self, "lora_down") or not isinstance(getattr(self, "lora_down"), torch.nn.Linear):
                            self.lora_down = torch.nn.Linear(self.base.out_features, self.rank, bias=False,
                                                             device=self.base.weight.device, dtype=self.base.weight.dtype)
                            torch.nn.init.normal_(self.lora_down.weight, mean=0.0, std=0.02)
                        if not hasattr(self, "lora_up") or not isinstance(getattr(self, "lora_up"), torch.nn.Linear):
                            self.lora_up = torch.nn.Linear(self.rank, self.base.out_features, bias=False,
                                                           device=self.base.weight.device, dtype=self.base.weight.dtype)
                            torch.nn.init.normal_(self.lora_up.weight, mean=0.0, std=0.02)
                    def forward(self, x):
                        z = self.base(x)
                        return z + self.lora_up(self.lora_down(z)) * self.scaling
                setattr(_main, "ActivationSpaceLoRAWrapper", ActivationSpaceLoRAWrapper)
    except Exception:
        # If transformers import fails for some reason, fall back to default load
        pass

    loaded_obj = torch.load(model_id, weights_only=False, map_location='cpu')

    tokenizer = None
    model = None
    base_model_hint = None
    if isinstance(loaded_obj, dict) and 'model' in loaded_obj:
        tokenizer = loaded_obj.get('tokenizer')
        model = loaded_obj['model']
        base_model_hint = (
            loaded_obj.get('base_model')
            or loaded_obj.get('model_id')
            or loaded_obj.get('tokenizer_model')
        )
    else:
        model = loaded_obj

    if model is None:
        raise ValueError(f"Unsupported local checkpoint format: {model_id}")

    # Ensure config compatibility across transformers versions
    try:
        cfg = model.config
        # Set defaults for private flags used by properties in recent versions
        def _ensure(prop, default):
            try:
                # Access to trigger AttributeError if backing field missing
                _ = getattr(cfg, prop)
            except Exception:
                try:
                    setattr(cfg, prop, default)
                except Exception:
                    # Fallback to private backing name that properties expect
                    setattr(cfg, f"_{prop}", default)

        for prop, default in (
            ("output_attentions", False),
            ("output_hidden_states", False),
            ("return_dict", True),
            ("use_return_dict", True),
            ("use_cache", False),
        ):
            _ensure(prop, default)

        # Provide a reasonable sequence length hint
        if not hasattr(model, 'seqlen'):
            try:
                model.seqlen = getattr(cfg, 'max_position_embeddings', 2048)
            except Exception:
                model.seqlen = 2048
        # LlamaModel in newer Transformers expects a shared rotary embedding module on the model
        try:
            inner = getattr(model, 'model', None)
            if inner is not None and not hasattr(inner, 'rotary_emb') and getattr(cfg, 'model_type', '') == 'llama':
                try:
                    # Prefer HF's implementation to match version-specific behavior
                    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFLlamaRotaryEmbedding
                    inner.rotary_emb = HFLlamaRotaryEmbedding(config=cfg)
                except Exception:
                    # Fallback to local implementation with basic defaults
                    from component.svd_llama import LlamaRotaryEmbedding as LocalLlamaRotaryEmbedding
                    head_dim = cfg.hidden_size // cfg.num_attention_heads
                    max_pos = getattr(cfg, 'max_position_embeddings', 2048)
                    inner.rotary_emb = LocalLlamaRotaryEmbedding(head_dim, max_position_embeddings=max_pos)
        except Exception:
            pass
    except Exception:
        pass

    # Align dtype with the HF loading path which defaults to float16 for speed/memory
    try:
        model = model.to(dtype=torch.float16)
    except Exception:
        # If some submodules cannot change dtype (e.g., layernorm buffers), skip silently
        pass

    tokenizer = _recover_tokenizer_from_model(
        model,
        tokenizer=tokenizer,
        explicit_hint=base_model_hint,
        hf_token=hf_token,
        checkpoint_path=model_id,
    )

    return model, tokenizer


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
