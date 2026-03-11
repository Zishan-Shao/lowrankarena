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
        has_min_api = (
            hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode")
        ) or (
            hasattr(tokenizer, "pad_token_id") and hasattr(tokenizer, "eos_token_id")
        ) or (
            hasattr(tokenizer, "vocab_size") or hasattr(tokenizer, "get_vocab")
        )
        return bool(has_min_api)
    except Exception:
        return False

def _load_tokenizer_from_hint(model_hint: str, hf_token: str = None):
    if not model_hint:
        return None
    hf_token = _resolve_hf_token(hf_token)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            model_hint, trust_remote_code=True, use_fast=True, token=hf_token
        )
        if _tokenizer_ok(tok):
            return tok
    except Exception:
        pass
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            model_hint, trust_remote_code=True, use_fast=False, token=hf_token
        )
        if _tokenizer_ok(tok):
            return tok
    except Exception:
        pass
    return None

def _checkpoint_path_model_hints(model_path: str):
    if not model_path:
        return []
    lower = str(model_path).lower()
    hints = []
    explicit_map = {
        "jeffwan_llama_7b_hf": "jeffwan/llama-7b-hf",
        "jeffwan_llama_13b_hf": "jeffwan/llama-13b-hf",
        "jeffwan_llama_30b_hf": "jeffwan/llama-30b-hf",
        "openlm_research_open_llama_7b": "openlm-research/open_llama_7b",
    }
    for needle, hint in explicit_map.items():
        if needle in lower and hint not in hints:
            hints.append(hint)
    return hints

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

def get_model_from_local(model_id):
    """
    Load a locally saved checkpoint produced by this repo.

    Older checkpoints were saved with full Python objects:
        torch.save({'model': model, 'tokenizer': tokenizer}, path)

    Unpickling those requires the same transformers symbols to exist at load
    time. Some versions no longer expose transformers.activations.SiLUActivation,
    which raises an AttributeError while loading.

    To maximize compatibility without forcing a specific transformers version,
    we provide a lightweight shim for missing activations before torch.load.
    """
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
        # LLaMA attention shim: some checkpoints were saved with Transformers that defined LlamaSdpaAttention
        try:
            import transformers.models.llama.modeling_llama as _llama_mod
            if not hasattr(_llama_mod, "LlamaSdpaAttention") and hasattr(_llama_mod, "LlamaAttention"):
                class LlamaSdpaAttention(_llama_mod.LlamaAttention):
                    pass
                setattr(_llama_mod, "LlamaSdpaAttention", LlamaSdpaAttention)
        except Exception:
            pass

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
    if isinstance(loaded_obj, dict) and 'model' in loaded_obj:
        tokenizer = loaded_obj.get('tokenizer')
        model = loaded_obj['model']
    else:
        # Newer checkpoints may be saved as a bare model object.
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

    # If tokenizer was not serialized, reconstruct it from explicit env hint first.
    if not _tokenizer_ok(tokenizer):
        model_hint = (
            os.getenv("SVDLLM_TOKENIZER_MODEL")
            or getattr(model, "name_or_path", None)
            or getattr(getattr(model, "config", None), "_name_or_path", None)
            or getattr(getattr(model, "config", None), "name_or_path", None)
        )
        hints = [h for h in [model_hint, *_checkpoint_path_model_hints(model_id)] if h]
        if not hints:
            raise ValueError(
                "Checkpoint does not include tokenizer; set SVDLLM_TOKENIZER_MODEL "
                "to a valid local tokenizer path or base model id."
            )
        for hint in hints:
            tokenizer = _load_tokenizer_from_hint(hint)
            if _tokenizer_ok(tokenizer):
                break

    if not _tokenizer_ok(tokenizer):
        # Final fallback for llama-family checkpoints.
        tokenizer = _load_tokenizer_from_hint("openlm-research/open_llama_7b")
    if not _tokenizer_ok(tokenizer):
        raise TypeError("Tokenizer recovery failed. Set SVDLLM_TOKENIZER_MODEL to a valid tokenizer/base model id.")

    # Ensure pad token exists for batched evals.
    try:
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        pass

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
