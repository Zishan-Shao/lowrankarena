from transformers.models.llama.configuration_llama import LlamaConfig


class BasisSharingLlamaConfig(LlamaConfig):
    model_type = "basis_sharing_llama"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
