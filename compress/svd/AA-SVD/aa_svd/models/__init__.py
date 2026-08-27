import logging
from typing import Dict, Any

import torch
from omegaconf import DictConfig
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForImageClassification,
    AutoTokenizer,
)

from ..utils import safe_pop

logger = logging.getLogger(__name__)


def create_model(cfg: DictConfig) -> Dict[str, Any]:
    """Factory function to create models based on configuration."""
    cfg = cfg.copy()
    model_type = safe_pop(cfg, "type", None)
    if model_type == "hf_pretrained":
        return load_hf_pretrained(cfg)
    elif model_type == "hf_from_scratch":
        return load_hf_from_scratch(cfg)
    elif model_type == "timm_pretrained":
        return load_timm_pretrained(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_timm_pretrained(cfg: DictConfig) -> Dict[str, Any]:
    """Load a pretrained model from timm."""
    import timm
    logger.info(f"Loading pretrained model {cfg.name} from timm")

    task_type = safe_pop(cfg, "task", "image_classification")
    if task_type != "image_classification":
        raise NotImplementedError(
            "Timm model loading currently only supports image classification."
        )

    model = timm.create_model(cfg.name, pretrained=True)

    if cfg.get("load_tokenizer", False):
        logger.warning(
            "Timm models do not have associated tokenizers. "
            "Skipping tokenizer loading."
        )

    return {"model": model, "tokenizer": None}


def load_hf_pretrained(cfg: DictConfig) -> Dict[str, Any]:
    """Load a pretrained model from Hugging Face."""
    logger.info(f"Loading pretrained model {cfg.name} from Hugging Face")

    model_cls = get_hf_model_class(safe_pop(cfg, "task", "base"))
    model = model_cls.from_pretrained(
        cfg.name,
        cache_dir=cfg.get("cache_dir"),
        revision=cfg.get("revision", "main"),
        torch_dtype=getattr(torch, cfg.get("dtype", "float32")),
        device_map=cfg.get("device_map", "auto"),
    )

    tokenizer = None
    if cfg.get("load_tokenizer", True):
        logger.info(f"Loading tokenizer for {cfg.name}")
        if 'jeffwan' in cfg.name:
            # jeffwan/llama models predate AutoTokenizer support
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(
                cfg.name, trust_remote_code=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.name,
                cache_dir=cfg.get("cache_dir"),
                revision=cfg.get("revision", "main"),
                use_fast=cfg.get("use_fast_tokenizer", True),
            )

    return {"model": model, "tokenizer": tokenizer}


def load_hf_from_scratch(cfg: DictConfig) -> Dict[str, Any]:
    """Initialize a model from scratch using a Hugging Face architecture."""
    from transformers import AutoConfig

    logger.info(f"Initializing model {cfg.name} from scratch")

    model_cls = get_hf_model_class(safe_pop(cfg, "task", "base"))
    logger.info(f"Loading model class {model_cls}")

    config = AutoConfig.from_pretrained(
        cfg.name,
        device_map=cfg.get("device_map", "auto"),
        cache_dir=cfg.get("cache_dir"),
        revision=cfg.get("revision", "main"),
    )
    logger.info(f"Loaded config {config} for model {cfg.name}")
    model = model_cls.from_config(
        config,
        torch_dtype=getattr(torch, cfg.get("dtype", "float32")),
    )

    tokenizer = None
    if cfg.get("load_tokenizer", True):
        logger.info(f"Loading tokenizer for {cfg.name}")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.name,
            cache_dir=cfg.get("cache_dir"),
            use_fast=cfg.get("use_fast_tokenizer", True),
        )

    return {"model": model, "tokenizer": tokenizer}


def get_hf_model_class(task: str):
    """Get the appropriate Hugging Face model class for the given task."""
    if task == "base":
        return AutoModel
    elif task == "causal_lm":
        return AutoModelForCausalLM
    elif task == "seq_classification":
        return AutoModelForSequenceClassification
    elif task == "image_classification":
        return AutoModelForImageClassification
    else:
        raise NotImplementedError(
            f"Model class for task '{task}' is not implemented."
        )
