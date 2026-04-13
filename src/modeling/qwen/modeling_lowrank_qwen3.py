from __future__ import annotations

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model

try:
    from src.modeling.common import apply_low_rank_replacements
except ImportError:  # pragma: no cover - used when copied into a standalone HF artifact.
    from .common import apply_low_rank_replacements

from .configuration_lowrank_qwen3 import LowRankQwen3Config


class LowRankQwen3Model(Qwen3Model):
    config_class = LowRankQwen3Config
    _supports_attention_backend = True

    def __init__(self, config):
        super().__init__(config)
        self.replaced_low_rank_modules, self.missing_low_rank_modules = apply_low_rank_replacements(
            self,
            getattr(config, "low_rank_modules", {}) or {},
            strict=True,
        )


class LowRankQwen3ForCausalLM(Qwen3ForCausalLM):
    config_class = LowRankQwen3Config

    def __init__(self, config):
        super().__init__(config)
        self.model = LowRankQwen3Model(config)
