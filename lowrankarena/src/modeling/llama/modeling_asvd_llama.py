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
        truncation_ranks = dict(getattr(config, "truncation_ranks", {}) or {})
        truncation_ranks.pop("lm_head", None)
        self.replaced_low_rank_modules, self.missing_low_rank_modules = apply_low_rank_replacements(
            self,
            truncation_ranks,
            strict=True,
        )


class ASVDLlamaForCausalLM(LlamaForCausalLM):
    config_class = ASVDLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.replaced_low_rank_modules, self.missing_low_rank_modules = apply_low_rank_replacements(
            self,
            getattr(config, "truncation_ranks", {}) or {},
            strict=True,
        )
