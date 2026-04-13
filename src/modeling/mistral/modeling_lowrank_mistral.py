from __future__ import annotations

from transformers.models.mistral.modeling_mistral import MistralForCausalLM, MistralModel

try:
    from src.modeling.common import apply_low_rank_replacements
except ImportError:  # pragma: no cover - used when copied into a standalone HF artifact.
    from .common import apply_low_rank_replacements

from .configuration_lowrank_mistral import LowRankMistralConfig


class LowRankMistralModel(MistralModel):
    config_class = LowRankMistralConfig
    _supports_attention_backend = True

    def __init__(self, config):
        super().__init__(config)
        self.replaced_low_rank_modules, self.missing_low_rank_modules = apply_low_rank_replacements(
            self,
            getattr(config, "low_rank_modules", {}) or {},
            strict=True,
        )


class LowRankMistralForCausalLM(MistralForCausalLM):
    config_class = LowRankMistralConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LowRankMistralModel(config)
