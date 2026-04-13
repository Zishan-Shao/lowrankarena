from __future__ import annotations

from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model

try:
    from src.modeling.common import apply_low_rank_replacements
except ImportError:  # pragma: no cover - used when copied into a standalone HF artifact.
    from .common import apply_low_rank_replacements

from .configuration_lowrank_qwen2 import LowRankQwen2Config


class LowRankQwen2Model(Qwen2Model):
    config_class = LowRankQwen2Config
    _supports_attention_backend = True

    def __init__(self, config):
        super().__init__(config)
        self.replaced_low_rank_modules, self.missing_low_rank_modules = apply_low_rank_replacements(
            self,
            getattr(config, "low_rank_modules", {}) or {},
            strict=True,
        )


class LowRankQwen2ForCausalLM(Qwen2ForCausalLM):
    config_class = LowRankQwen2Config

    def __init__(self, config):
        super().__init__(config)
        self.model = LowRankQwen2Model(config)
