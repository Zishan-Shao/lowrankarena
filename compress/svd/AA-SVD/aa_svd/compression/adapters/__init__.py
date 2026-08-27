from transformers import LlamaForCausalLM, Qwen2ForCausalLM

from .llama_adapter import LlamaLayerAdapter, LlamaHeadAdapter, LlamaModelAdapter
from .qwen2_adapter import Qwen2LayerAdapter, Qwen2HeadAdapter, Qwen2ModelAdapter

MODEL_ADAPTER_REGISTRY = {
    LlamaForCausalLM: LlamaModelAdapter,
    Qwen2ForCausalLM: Qwen2ModelAdapter,
}

__all__ = [
    "LlamaLayerAdapter",
    "LlamaHeadAdapter",
    "LlamaModelAdapter",
    "Qwen2LayerAdapter",
    "Qwen2HeadAdapter",
    "Qwen2ModelAdapter",
    "MODEL_ADAPTER_REGISTRY",
]

