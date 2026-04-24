from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class LowRankQwen3Config(Qwen3Config):
    model_type = "lowrank_qwen3"

    def __init__(
        self,
        low_rank_modules=None,
        low_rank_method="generic_low_rank",
        low_rank_schema="ABLinear",
        low_rank_format_version=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.low_rank_modules = low_rank_modules or {}
        self.low_rank_method = low_rank_method
        self.low_rank_schema = low_rank_schema
        self.low_rank_format_version = int(low_rank_format_version)
