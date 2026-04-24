from transformers.models.llama.configuration_llama import LlamaConfig


class ASVDLlamaConfig(LlamaConfig):
    model_type = "asvd_llama"

    def __init__(self, truncation_ranks=None, **kwargs):
        super().__init__(**kwargs)
        self.truncation_ranks = truncation_ranks or {}
