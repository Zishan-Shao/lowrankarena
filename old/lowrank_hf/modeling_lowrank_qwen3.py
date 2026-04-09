from __future__ import annotations

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3Model,
)

from .configuration_lowrank_qwen3 import LowRankQwen3Config
from .modeling_lowrank_common import apply_low_rank_replacements


class LowRankQwen3Model(Qwen3Model):
    config_class = LowRankQwen3Config
    _supports_attention_backend = True

    def __init__(self, config):
        super().__init__(config)
        apply_low_rank_replacements(self, getattr(config, "low_rank_modules", {}) or {})


class LowRankQwen3ForCausalLM(Qwen3ForCausalLM):
    config_class = LowRankQwen3Config

    def __init__(self, config):
        super().__init__(config)
        self.model = LowRankQwen3Model(config)
