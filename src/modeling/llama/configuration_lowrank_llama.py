from transformers.models.llama.configuration_llama import LlamaConfig


class LowRankLlamaConfig(LlamaConfig):
    model_type = "lowrank_llama"

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
