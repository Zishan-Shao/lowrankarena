### Non-MoE Qwen3

from .LlamaAdapter import LlamaAdapter


class QwenAdapter(LlamaAdapter):
    @property
    def arch(self) -> str:
        return "qwen3"
