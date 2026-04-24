from __future__ import annotations

from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel

try:
    from src.modeling.common import apply_low_rank_replacements
except ImportError:  # pragma: no cover - used when copied into a standalone HF artifact.
    from .common import apply_low_rank_replacements

from .configuration_lowrank_llama import LowRankLlamaConfig


class LowRankLlamaModel(LlamaModel):
    config_class = LowRankLlamaConfig
    _supports_attention_backend = True

    def __init__(self, config):
        super().__init__(config)
        self.replaced_low_rank_modules, self.missing_low_rank_modules = apply_low_rank_replacements(
            self,
            getattr(config, "low_rank_modules", {}) or {},
            strict=True,
        )


class LowRankLlamaForCausalLM(LlamaForCausalLM):
    config_class = LowRankLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LowRankLlamaModel(config)
