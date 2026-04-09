from __future__ import annotations

from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
)

from .configuration_lowrank_llama import LowRankLlamaConfig
from .modeling_lowrank_common import apply_low_rank_replacements


class LowRankLlamaModel(LlamaModel):
    config_class = LowRankLlamaConfig
    _supports_attention_backend = True

    def __init__(self, config):
        super().__init__(config)
        apply_low_rank_replacements(self, getattr(config, "low_rank_modules", {}) or {})


class LowRankLlamaForCausalLM(LlamaForCausalLM):
    config_class = LowRankLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LowRankLlamaModel(config)
