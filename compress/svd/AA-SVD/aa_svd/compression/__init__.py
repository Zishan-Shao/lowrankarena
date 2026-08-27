import json
import logging
from typing import Dict, Union

import torch
import torch.nn as nn
import wandb
from transformers import PreTrainedModel

from ..utils import get_device
from .compress import apply_compression_parallel

logger = logging.getLogger(__name__)


def apply_compression(
    model: Union[PreTrainedModel, nn.Module],
    config: Dict,
    **kwargs
) -> nn.Module:
    """Apply SVD-based compression to all linear layers of a model."""

    model = model.to('cpu')
    torch.cuda.empty_cache()

    if hasattr(model, 'config'):
        use_cache_original = model.config.use_cache
        model.config.use_cache = False

    sub_method = getattr(config, 'sub_method', None)
    assert sub_method is not None, "sub_method must be specified in config"

    device = getattr(config, 'device', get_device())

    if sub_method == 'no-compress':
        return model.to(device)

    calibration_dataloader_train = kwargs.get('calibration_dataloader_train')
    calibration_dataloader_val = kwargs.get('calibration_dataloader_val')

    # include_only and exclude_only are mutually exclusive filters
    include_only = getattr(config, 'include_only', None)
    exclude_modules = getattr(config, 'exclude_only', None)

    if include_only is not None and exclude_modules is not None:
        raise ValueError(
            "Cannot specify both 'include_only' and 'exclude_modules'"
        )

    all_modules = [
        (name, mod) for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    ]

    if include_only is not None:
        if isinstance(include_only, str):
            include_only = [include_only]
        all_modules = [(n, m) for n, m in all_modules if n in include_only]
    elif exclude_modules is not None:
        if isinstance(exclude_modules, str):
            exclude_modules = [exclude_modules]
        all_modules = [
            (n, m) for n, m in all_modules if n not in exclude_modules
        ]

    logger.info(f"Found {len(all_modules)} linear layers for compression")

    if len(all_modules) == 0:
        raise ValueError(
            "No linear layers found for compression. "
            "Please check your configuration."
        )

    modules_to_replace = [name for name, _ in all_modules]

    target_param_ratio = getattr(config, 'target_param_ratio', 0.3)
    total_params_before = sum(p.numel() for p in model.parameters())

    # per-layer rank allocations
    rank_allocation_file_path = getattr(config, 'rank_allocation_file_path', None)
    if rank_allocation_file_path is not None:
        with open(rank_allocation_file_path, 'r') as f:
            allocations = json.load(f)
    else:
        allocations = {
            name: target_param_ratio for name in modules_to_replace
        }

    logger.info(f"Layerwise allocations: {json.dumps(allocations, indent=2)}")

    compressed_model = apply_compression_parallel(
        config, model, modules_to_replace,
        calibration_dataloader=calibration_dataloader_train,
        test_calibration_dataloader=calibration_dataloader_val,
        allocations=allocations,
        device=device,
    )

    del model
    torch.cuda.empty_cache()

    compressed_model = compressed_model.to(device)
    total_params_after = sum(p.numel() for p in compressed_model.parameters())
    ratio = total_params_after / total_params_before

    logger.info({
        'before': total_params_before,
        'after': total_params_after,
        'compression_ratio': ratio,
    })

    if wandb.run is not None:
        wandb.log({
            'params/before': total_params_before,
            'params/after': total_params_after,
            'params/compression_ratio': ratio,
        })

    if hasattr(compressed_model, 'config'):
        compressed_model.config.use_cache = use_cache_original

    return compressed_model


