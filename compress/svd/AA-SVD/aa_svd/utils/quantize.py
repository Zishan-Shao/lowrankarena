"""
LLM Quantization Toolkit (Hybrid)
===================================
Quantize HuggingFace models, including models with custom modules like
CompressedLinear (low-rank U/V factors stored as nn.Linear).

Strategy:
  - BnB methods (int8, nf4, fp4): In-place module replacement on the live model.
    This avoids save/reload issues with custom architectures where BnB's loading
    pipeline expects quantized weight metadata (SCB) that doesn't exist yet.
  - GPTQ: Save-then-reload via from_pretrained, which hooks into HF's quantization
    pipeline and produces properly quantized checkpoints with optimized kernels.

Supported methods:
  - gptq     : GPTQ 4/3/2-bit (requires calibration data, save-reload)
  - bnb-int8 : bitsandbytes LLM.int8() (in-place)
  - bnb-nf4  : bitsandbytes NormalFloat4 (in-place)
  - bnb-fp4  : bitsandbytes FP4 (in-place)

Usage:
    from llm_quantize import quantize_model

    # BnB on a live model (works with CompressedLinear)
    q_model = quantize_model(
        model=my_compressed_model,
        method="bnb-nf4",
        config={"compute_dtype": "bfloat16", "double_quant": True},
    )

    # GPTQ on a live model (saves to temp dir, reloads)
    q_model, tok = quantize_model(
        model=my_model,
        tokenizer=my_tokenizer,
        method="gptq",
        config={"bits": 4, "group_size": 128},
        calibration_data=cal_loader,
    )

    # GPTQ from a model_id directly
    q_model, tok = quantize_model(
        model_id="meta-llama/Llama-2-7b-hf",
        method="gptq",
        config={"bits": 4},
        calibration_data=cal_loader,
    )
"""

import gc
import logging
import os
import shutil
import tempfile
import time
from typing import Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default configs per method (hydra-style: override any key)
# ---------------------------------------------------------------------------
DEFAULT_CONFIGS = {
    "gptq": {
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
        "sym": True,
        "dataset": None,
    },
    "bnb-int8": {
        "threshold": 6.0,
        "skip_modules": None,
    },
    "bnb-nf4": {
        "compute_dtype": "bfloat16",
        "double_quant": False,
        "skip_modules": None,
    },
    "bnb-fp4": {
        "compute_dtype": "bfloat16",
        "double_quant": False,
        "skip_modules": None,
    },
}


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name in mapping:
        return mapping[name]
    raise ValueError(f"Unknown dtype '{name}'. Use one of {list(mapping.keys())}")


def _merge_config(method: str, overrides: Optional[dict]) -> dict:
    if method not in DEFAULT_CONFIGS:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Supported: {list(DEFAULT_CONFIGS.keys())}"
        )
    cfg = DEFAULT_CONFIGS[method].copy()
    if overrides:
        # Support OmegaConf DictConfig: convert to plain dict
        try:
            from omegaconf import OmegaConf, DictConfig
            if isinstance(overrides, DictConfig):
                overrides = OmegaConf.to_container(overrides, resolve=True)
        except ImportError:
            pass

        if not isinstance(overrides, dict):
            overrides = dict(overrides)

        unknown = set(overrides.keys()) - set(cfg.keys())
        if unknown:
            logger.warning(f"Unknown config keys for '{method}': {unknown} (ignored)")
        cfg.update({k: v for k, v in overrides.items() if k in cfg})
    return cfg


# ---------------------------------------------------------------------------
# Module discovery and replacement helpers
# ---------------------------------------------------------------------------
def _find_linear_modules(
    model: nn.Module,
    skip_modules: Optional[list] = None,
) -> dict:
    """
    Walk the model tree and find all nn.Linear modules.
    For CompressedLinear layers, this discovers the inner U and V factors.
    Returns dict of {fully_qualified_name: module}.
    """
    skip = set(skip_modules or [])
    linears = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(s in name for s in skip):
                continue
            linears[name] = module
    return linears


def _get_parent_and_attr(model: nn.Module, fqn: str):
    """Split 'a.b.c.d' into (model.a.b.c, 'd') for setattr replacement."""
    parts = fqn.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


# ---------------------------------------------------------------------------
# BnB in-place quantization
# ---------------------------------------------------------------------------
def _quantize_bnb_int8(model: nn.Module, cfg: dict) -> nn.Module:
    """Replace nn.Linear modules with bnb.nn.Linear8bitLt in-place."""
    import bitsandbytes as bnb

    target_modules = _find_linear_modules(model, cfg["skip_modules"])
    logger.info(f"BnB int8: quantizing {len(target_modules)} Linear modules")

    device = next(model.parameters()).device
    count = 0
    for fqn, linear in target_modules.items():
        parent, attr = _get_parent_and_attr(model, fqn)
        has_bias = linear.bias is not None

        new_module = bnb.nn.Linear8bitLt(
            linear.in_features,
            linear.out_features,
            bias=has_bias,
            has_fp16_weights=False,
            threshold=cfg["threshold"],
        )

        # Transfer weights — must be fp16 for bnb
        new_module.weight = bnb.nn.Int8Params(
            linear.weight.data.to(torch.float16).contiguous(),
            requires_grad=False,
            has_fp16_weights=False,
        )
        if has_bias:
            new_module.bias = nn.Parameter(
                linear.bias.data.to(torch.float16),
                requires_grad=False,
            )

        # Move to device to trigger quantization
        new_module = new_module.to(device)
        setattr(parent, attr, new_module)
        count += 1

    logger.info(f"BnB int8: replaced {count} modules")
    return model


def _quantize_bnb_4bit(model: nn.Module, cfg: dict, quant_type: str) -> nn.Module:
    """Replace nn.Linear modules with bnb.nn.Linear4bit in-place."""
    import bitsandbytes as bnb

    compute_dtype = _resolve_dtype(cfg["compute_dtype"])
    target_modules = _find_linear_modules(model, cfg["skip_modules"])
    logger.info(
        f"BnB 4-bit ({quant_type}): quantizing {len(target_modules)} Linear modules"
    )

    device = next(model.parameters()).device
    count = 0
    for fqn, linear in target_modules.items():
        parent, attr = _get_parent_and_attr(model, fqn)
        has_bias = linear.bias is not None

        new_module = bnb.nn.Linear4bit(
            linear.in_features,
            linear.out_features,
            bias=has_bias,
            compute_dtype=compute_dtype,
            quant_type=quant_type,
            compress_statistics=cfg["double_quant"],
        )

        # Transfer weights — must be fp16 for bnb
        new_module.weight = bnb.nn.Params4bit(
            linear.weight.data.to(torch.float16).contiguous(),
            requires_grad=False,
            quant_type=quant_type,
            compress_statistics=cfg["double_quant"],
        )
        if has_bias:
            new_module.bias = nn.Parameter(
                linear.bias.data.to(torch.float16),
                requires_grad=False,
            )

        # Move to device to trigger quantization
        new_module = new_module.to(device)
        setattr(parent, attr, new_module)
        count += 1

    logger.info(f"BnB 4-bit ({quant_type}): replaced {count} modules")
    return model


# ---------------------------------------------------------------------------
# GPTQ quantization (save-reload via from_pretrained)
# ---------------------------------------------------------------------------
def _calibration_texts_from_loader(
    calibration_data: DataLoader,
    tokenizer,
    max_samples: int = 256,
) -> list[str]:
    """Decode tokenised calibration batches back to strings for GPTQConfig."""
    texts = []
    for batch in calibration_data:
        if isinstance(batch, dict):
            ids = batch.get("input_ids", None)
        elif isinstance(batch, (list, tuple)):
            ids = batch[0]
        else:
            ids = batch
        if ids is None:
            continue
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        for row in ids:
            texts.append(tokenizer.decode(row, skip_special_tokens=True))
            if len(texts) >= max_samples:
                return texts
    return texts


def _quantize_gptq(
    model: Optional[nn.Module],
    model_id: Optional[str],
    tokenizer,
    calibration_data: Optional[DataLoader],
    cfg: dict,
    device_map: str,
    torch_dtype,
):
    """
    GPTQ quantization via optimum's GPTQQuantizer directly.

    When a live model is passed (e.g. a compressed model with CompressedLinear
    layers), GPTQ is run directly on it so that U and V are each quantized as
    independent nn.Linear layers — preserving the low-rank structure while
    reducing each factor to the target bit-width.

    When only model_id is passed, the model is loaded fresh from disk via
    AutoModelForCausalLM.from_pretrained.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from optimum.gptq import GPTQQuantizer

    tmp_dir = None
    try:
        # Resolve calibration data (needed regardless of which path we take)
        if calibration_data is not None:
            if tokenizer is None:
                raise ValueError("tokenizer required when calibration_data is provided")
            dataset = _calibration_texts_from_loader(calibration_data, tokenizer)
            logger.info(f"GPTQ: using {len(dataset)} calibration samples")
        elif cfg["dataset"] is not None:
            dataset = cfg["dataset"]
            logger.info(f"GPTQ: using named dataset '{dataset}'")
        else:
            dataset = "c4"
            logger.info("GPTQ: no calibration data, falling back to 'c4'")

        quantizer = GPTQQuantizer(
            bits=cfg["bits"],
            group_size=cfg["group_size"],
            desc_act=cfg["desc_act"],
            sym=cfg["sym"],
            dataset=dataset,
        )

        if model is not None:
            # Run GPTQ directly on the live model.
            # This preserves custom architectures (e.g. CompressedLinear): the
            # quantizer treats every nn.Linear it finds — including U and V
            # sub-layers — as independent layers to quantize.
            logger.info(
                f"GPTQ: quantizing live model in-place with "
                f"bits={cfg['bits']}, group_size={cfg['group_size']}"
            )

            # Diagnostic: log any nn.Linear with a degenerate (0-sized) weight.
            # gptqmodel crashes with IndexError if columns==0; skip those layers
            # by replacing them with a no-op so quantization can proceed.
            for name, module in list(model.named_modules()):
                if isinstance(module, nn.Linear):
                    if 0 in module.weight.shape:
                        logger.warning(
                            f"Skipping degenerate nn.Linear '{name}' with "
                            f"weight shape {tuple(module.weight.shape)} — "
                            f"gptqmodel cannot quantize 0-sized dimensions."
                        )
                        # Replace with an identity-preserving stub so the rest
                        # of the graph is unaffected during calibration.
                        stub = nn.Linear(
                            module.in_features, module.out_features,
                            bias=module.bias is not None,
                        )
                        stub.weight.data.zero_()
                        if module.bias is not None:
                            stub.bias.data.copy_(module.bias.data)
                        parent_name, _, child_name = name.rpartition(".")
                        parent = model.get_submodule(parent_name) if parent_name else model
                        setattr(parent, child_name, stub)
                    else:
                        logger.info(
                            f"nn.Linear '{name}' weight shape {tuple(module.weight.shape)}"
                        )

            if not hasattr(model, "hf_device_map"):
                model.hf_device_map = {}
            q_model = quantizer.quantize_model(model, tokenizer)
            return q_model, tokenizer

        # No live model — load from model_id
        if model_id is None:
            raise ValueError("Either model or model_id required for GPTQ.")

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_id)

        logger.info(
            f"GPTQ: loading from '{model_id}' with "
            f"bits={cfg['bits']}, group_size={cfg['group_size']}"
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        if not hasattr(base_model, "hf_device_map"):
            base_model.hf_device_map = {}

        q_model = quantizer.quantize_model(base_model, tokenizer)
        return q_model, tokenizer

    finally:
        if tmp_dir is not None and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def quantize_model(
    model: Optional[nn.Module] = None,
    model_id: Optional[str] = None,
    tokenizer=None,
    method: str = "bnb-nf4",
    config: Optional[dict] = None,
    calibration_data: Optional[DataLoader] = None,
    device_map: str = "auto",
    torch_dtype=torch.float16,
) -> Union[nn.Module, tuple]:
    """
    Quantize a HuggingFace model.

    - BnB methods (bnb-int8, bnb-nf4, bnb-fp4): Applied in-place on the live
      model. Works with custom architectures like CompressedLinear. Returns the
      modified model directly.

    - GPTQ: Uses HF's from_pretrained pipeline (save-reload). Returns
      (model, tokenizer) tuple.

    Parameters
    ----------
    model : nn.Module, optional
        A live model to quantize. Required for BnB methods.
    model_id : str, optional
        HuggingFace model ID or local path. Can be used instead of model
        for GPTQ, or in addition to model (ignored for BnB).
    tokenizer : PreTrainedTokenizer, optional
        Required for GPTQ.
    method : str
        "gptq", "bnb-int8", "bnb-nf4", "bnb-fp4".
    config : dict or OmegaConf DictConfig, optional
        Method-specific overrides (hydra-style).
    calibration_data : DataLoader, optional
        Tokenised calibration data for GPTQ.
    device_map : str
        Device placement for GPTQ reload (default: "auto").
    torch_dtype : torch.dtype
        Base dtype for GPTQ reload (default: float16).

    Returns
    -------
    nn.Module or (nn.Module, tokenizer)
        BnB methods: returns the quantized model.
        GPTQ: returns (quantized_model, tokenizer) tuple.
    """
    cfg = _merge_config(method, config)

    start_time = time.time()

    if method in ("bnb-int8", "bnb-nf4", "bnb-fp4"):
        # --- In-place BnB quantization ---
        if model is None:
            raise ValueError(f"'{method}' requires a live model. Pass model=...")

        all_linears = _find_linear_modules(model, cfg.get("skip_modules"))
        n_params = sum(m.weight.numel() for m in all_linears.values())
        logger.info(
            f"Quantizing with {method}: "
            f"{len(all_linears)} Linear modules, {n_params:,} parameters"
        )

        if method == "bnb-int8":
            model = _quantize_bnb_int8(model, cfg)
        elif method == "bnb-nf4":
            model = _quantize_bnb_4bit(model, cfg, "nf4")
        elif method == "bnb-fp4":
            model = _quantize_bnb_4bit(model, cfg, "fp4")

        elapsed = time.time() - start_time
        logger.info(f"Quantization completed in {elapsed:.1f}s")
        return model, tokenizer

    elif method == "gptq":
        # --- GPTQ via save-reload ---
        q_model, tok = _quantize_gptq(
            model=model,
            model_id=model_id,
            tokenizer=tokenizer,
            calibration_data=calibration_data,
            cfg=cfg,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        elapsed = time.time() - start_time
        logger.info(f"GPTQ quantization completed in {elapsed:.1f}s")
        return q_model, tok

    else:
        raise ValueError(f"Unknown method '{method}'")