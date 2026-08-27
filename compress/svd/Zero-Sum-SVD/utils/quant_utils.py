"""
Quantization utilities for 8-bit weight compression using bitsandbytes.

This module provides functions for:
- Row-wise 8-bit quantization/dequantization using bitsandbytes
- In-place quantization of target linear modules (nn.Linear and LowRankLinear)
- Loading quantized checkpoints with automatic dequantization
- Quantize-dequantize roundtrip for evaluation consistency
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.model_utils import (
    LowRankLinear,
    get_layers,
    find_target_modules_in_layer,
)
from utils.correction_utils import get_module_from_key

# Optional 8-bit quantization via bitsandbytes
try:
    import bitsandbytes as bnb
except Exception:
    bnb = None


def _require_bnb():
    """Check that bitsandbytes is available and CUDA is present."""
    if bnb is None:
        raise RuntimeError("bitsandbytes is required for --quantize_8_bit")
    if not torch.cuda.is_available():
        raise RuntimeError("--quantize_8_bit requires CUDA for bitsandbytes quantization")


def bnb_quantize_rowwise(mat_2d: torch.Tensor, code: torch.Tensor):
    """
    Row-wise 8-bit quantization using bnb.functional.quantize on each row.

    Args:
        mat_2d: 2D tensor to quantize [rows, cols]
        code: Dynamic map code from bitsandbytes

    Returns:
        q_uint8: [rows, cols] uint8 quantized values
        absmax_fp16: [rows] float16, one scale per row
    """
    assert mat_2d.dim() == 2
    rows = mat_2d.size(0)
    q_rows = []
    absmax_rows = []

    x = mat_2d.contiguous()
    for r in range(rows):
        q, (absmax, _) = bnb.functional.quantize(x[r].float(), code=code)
        q_rows.append(q)
        absmax_rows.append(absmax)

    q_uint8 = torch.stack(q_rows, dim=0).to(torch.uint8)
    absmax_fp16 = torch.stack(absmax_rows, dim=0).to(torch.float16)
    return q_uint8, absmax_fp16


def bnb_dequantize_rowwise(q_uint8: torch.Tensor, absmax_fp16: torch.Tensor, code: torch.Tensor, out_dtype=torch.float16):
    """
    Inverse of bnb_quantize_rowwise, uses dequantize_no_absmax then rescales per row.

    Args:
        q_uint8: [rows, cols] uint8 quantized values
        absmax_fp16: [rows] float16 scales
        code: Dynamic map code from bitsandbytes
        out_dtype: Output dtype (default float16)

    Returns:
        Dequantized tensor in out_dtype
    """
    assert q_uint8.dim() == 2
    rows = q_uint8.size(0)
    out_rows = []

    q = q_uint8
    a = absmax_fp16.to(torch.float32)
    for r in range(rows):
        deq = bnb.functional.dequantize_no_absmax(q[r], code=code).to(torch.float32)
        deq = deq * a[r]
        out_rows.append(deq)

    out = torch.stack(out_rows, dim=0).to(out_dtype)
    return out


def quantize_target_modules_inplace(args, model, dev, verbose=True):
    """
    Quantizes only target linear modules:
      - nn.Linear in find_target_modules_in_layer
      - LowRankLinear factors u_proj and v_proj

    Mutates model weights to uint8 for storage.

    Args:
        args: Arguments namespace with model name
        model: The model to quantize
        dev: Device to use for quantization
        verbose: Whether to show progress bar

    Returns:
        mapping_info: Dict with absmax scales for each module (q is stored in model weights)
    """
    _require_bnb()
    if verbose:
        print("   🔢 Quantizing target modules to int8...")

    code = bnb.functional.create_dynamic_map().to(dev)
    layers = get_layers(args.model, model)

    mapping_info = {
        "meta": {
            "scheme": "bnb_rowwise_uint8",
            "code_kind": "dynamic_map",
        },
        "modules": {}
    }

    layer_iter = tqdm(layers, desc="Quantizing", total=len(layers), leave=False) if verbose else layers
    for layer_idx, layer in enumerate(layer_iter):
        subset = find_target_modules_in_layer(layer)
        for name, mod in subset.items():
            key = f"layer{layer_idx}.{name}"

            if isinstance(mod, LowRankLinear):
                u_w = mod.u_proj.weight.data.to(dev)  # [out_features, rank]
                v_w = mod.v_proj.weight.data.to(dev)  # [rank, in_features]

                # Both u_proj and v_proj use row-wise quantization
                u_q, u_abs = bnb_quantize_rowwise(u_w, code=code)
                v_q, v_abs = bnb_quantize_rowwise(v_w, code=code)

                mapping_info["modules"][key] = {
                    "kind": "lowrank",
                    "u_absmax": u_abs.detach().cpu(),  # [out_features]
                    "v_absmax": v_abs.detach().cpu(),  # [rank]
                    "u_requires_grad": bool(mod.u_proj.weight.requires_grad),
                    "v_requires_grad": bool(mod.v_proj.weight.requires_grad),
                }

                with torch.no_grad():
                    mod.u_proj.weight.requires_grad_(False)
                    mod.v_proj.weight.requires_grad_(False)
                    mod.u_proj.weight.data = u_q.to(mod.u_proj.weight.device)
                    mod.v_proj.weight.data = v_q.to(mod.v_proj.weight.device)

            elif isinstance(mod, nn.Linear):
                w = mod.weight.data.to(dev)
                w_q, w_abs = bnb_quantize_rowwise(w, code=code)

                # Only store absmax scale (q is in model weights)
                mapping_info["modules"][key] = {
                    "kind": "dense",
                    "w_absmax": w_abs.detach().cpu(),
                    "w_requires_grad": bool(mod.weight.requires_grad),
                }

                with torch.no_grad():
                    mod.weight.requires_grad_(False)
                    mod.weight.data = w_q.to(mod.weight.device)

            else:
                continue

    return mapping_info


def dequantize_target_modules_inplace(saved_dict, dev="cuda", verbose=True):
    """
    Restores target modules back to fp16 using mapping_info saved in the same pt file.
    Reads q from model weights, absmax from mapping_info.

    Args:
        saved_dict: Dict from torch.load() with keys: model, tokenizer, mapping_info, model_name
        dev: Device to use for dequantization
        verbose: Whether to show progress bar

    Returns:
        The dequantized model
    """
    _require_bnb()
    model = saved_dict["model"]
    mapping_info = saved_dict.get("mapping_info", None)
    if mapping_info is None:
        return model

    if verbose:
        print("   🔢 Dequantizing target modules back to fp16...")

    code = bnb.functional.create_dynamic_map().to(dev)

    modules_info = mapping_info.get("modules", {})
    module_iter = tqdm(modules_info.items(), desc="Dequantizing", leave=False) if verbose else modules_info.items()
    for key, payload in module_iter:
        mod, layer_idx, module_name = get_module_from_key(saved_dict["model_name"], model, key)

        if payload["kind"] == "dense":
            # Read q from model weights, absmax from mapping_info
            q_uint8 = mod.weight.data.to(dev).to(torch.uint8)
            w_abs = payload["w_absmax"].to(dev)
            w = bnb_dequantize_rowwise(q_uint8, w_abs, code=code, out_dtype=torch.float16)

            req = payload.get("w_requires_grad", True)
            with torch.no_grad():
                mod.weight.data = w.to(mod.weight.device)
                mod.weight.requires_grad_(bool(req))

        elif payload["kind"] == "lowrank":
            # Read q from model weights, absmax from mapping_info
            u_q_uint8 = mod.u_proj.weight.data.to(dev).to(torch.uint8)
            v_q_uint8 = mod.v_proj.weight.data.to(dev).to(torch.uint8)
            u_abs = payload["u_absmax"].to(dev)
            v_abs = payload["v_absmax"].to(dev)

            # Both u_proj and v_proj use row-wise dequantization
            u = bnb_dequantize_rowwise(u_q_uint8, u_abs, code=code, out_dtype=torch.float16)
            v = bnb_dequantize_rowwise(v_q_uint8, v_abs, code=code, out_dtype=torch.float16)

            u_req = payload.get("u_requires_grad", True)
            v_req = payload.get("v_requires_grad", True)

            with torch.no_grad():
                mod.u_proj.weight.data = u.to(mod.u_proj.weight.device)
                mod.v_proj.weight.data = v.to(mod.v_proj.weight.device)
                mod.u_proj.weight.requires_grad_(bool(u_req))
                mod.v_proj.weight.requires_grad_(bool(v_req))

    return model


def load_quantized_pt(model_pt_path: str, device: str = "cuda"):
    """
    Loader that matches the pt style used in this project.
    Restores fp16 weights in place using the saved mapping_info.

    Args:
        model_pt_path: Path to the .pt checkpoint file
        device: Device to load the model to

    Returns:
        (model, tokenizer) tuple with dequantized weights
    """
    try:
        saved = torch.load(model_pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        # older PyTorch versions do not have weights_only
        saved = torch.load(model_pt_path, map_location="cpu")
    model = saved["model"]
    tokenizer = saved["tokenizer"]

    if saved.get("quantize_8_bit", False):
        model = model.to(device)
        dequantize_target_modules_inplace(saved, dev=device)
        model.eval()
    else:
        model = model.to(device).eval()

    return model, tokenizer


def quant_dequant_roundtrip_targets_(args, model, dev):
    """
    If --quantize_8_bit is enabled:
      1) quantize target modules (mutates weights to uint8)
      2) immediately dequantize back (mutates weights back to fp16)
      3) cast the whole model to bfloat16 for continued compute

    This simulates the quantization error that will be present in the saved checkpoint.

    Args:
        args: Arguments namespace with quantize_8_bit flag and model name
        model: The model to roundtrip
        dev: Device to use

    Returns:
        mapping_info if quantization was performed, None otherwise
    """
    if not getattr(args, "quantize_8_bit", False):
        return None

    print("   🔢 Quant-dequant roundtrip: simulating int8 quantization error...")
    _require_bnb()
    model = model.to(dev)

    mapping_info = quantize_target_modules_inplace(args, model, dev, verbose=False)

    # dequantize back in place using the mapping we just produced
    tmp = {"model": model, "model_name": args.model, "mapping_info": mapping_info}
    dequantize_target_modules_inplace(tmp, dev=dev, verbose=False)

    # continue the pipeline in bfloat16 when quant mode is enabled
    model.to(dtype=torch.bfloat16)
    print("   ✓ Roundtrip complete, model in bfloat16")
    return mapping_info


def _intermediate_eval_dtype(args):
    """
    Returns the dtype to use for intermediate evaluations.
    Uses bfloat16 when quantize_8_bit is enabled, float16 otherwise.
    """
    return torch.bfloat16 if getattr(args, "quantize_8_bit", False) else torch.float16
