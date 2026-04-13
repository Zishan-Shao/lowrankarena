from __future__ import annotations

from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel

try:
    from src.modeling.common import apply_low_rank_replacements
except ImportError:  # pragma: no cover - used when copied into a standalone HF artifact.
    from .common import apply_low_rank_replacements

from .configuration_asvd_llama import ASVDLlamaConfig


class ASVDLlamaModel(LlamaModel):
    config_class = ASVDLlamaConfig
    _supports_attention_backend = True

    def __init__(self, config):
        super().__init__(config)
        apply_low_rank_replacements(self, getattr(config, "truncation_ranks", {}) or {})


class ASVDLlamaForCausalLM(LlamaForCausalLM):
    config_class = ASVDLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        apply_low_rank_replacements(self, getattr(config, "truncation_ranks", {}) or {})
