import random
import numpy as np
import torch
import functools
import os
import logging
from omegaconf import DictConfig, ListConfig, open_dict
import json
from pathlib import Path
    

logger = logging.getLogger(__name__)

def setup_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def safe_pop(cfg: DictConfig, key, default = None):
    """Safely pop a key from the config dictionary"""
    with open_dict(cfg):
        val = cfg.pop(key, default)

    return val

def get_device():
    """Get the device to use"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_logging(cfg):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, cfg.level.upper(), None),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
def handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)

def save_json(obj, filename):
    file = Path(filename)
    dumped = json.dumps(
            obj, indent=2, default=handle_non_serializable, ensure_ascii=False
        )
    file.open("w", encoding="utf-8").write(dumped)

def get_dtype(dtype_str: str):
    """Get the torch dtype from string"""
    if dtype_str is None:
        return None
    dtype_str = dtype_str.lower()
    if dtype_str == "float32" or dtype_str == "fp32":
        return torch.float32
    elif dtype_str == "float64" or dtype_str == "fp64":
        return torch.float64
    elif dtype_str == "float16" or dtype_str == "fp16":
        return torch.float16
    elif dtype_str == "bfloat16" or dtype_str == "bf16":
        return torch.bfloat16
    elif dtype_str == "int8":
        return torch.int8
    elif dtype_str == "int4":
        return torch.int4
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")