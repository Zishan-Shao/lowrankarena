# flashsvdffnv15.py — v1.5 FFN entry point, fp32 auto-cast fallback
import importlib.util
import os
import torch

_spec = importlib.util.spec_from_file_location(
    "_flashsvdffn_v15_impl",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "flashsvdffn", "flashsvdffn_v1.5.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_impl = _mod.flashsvd_ffn


def flashsvd_ffn_v15(P, V1, U2, V2, b1, b2, BL=64, BD=128, BH=64, BR1=32, BR2=32):
    """v1.5 FFN with fp32 auto-cast fallback.

    When the model runs in fp16/bf16, tensors arrive at native dtype → zero
    cast overhead.  When running fp32 (e.g. --dtype fp32), we cast to fp16
    for the kernel and restore dtype on output.
    """
    orig = P.dtype
    if orig == torch.float32:
        # fallback: cast to fp16 for kernel, restore dtype on output
        P, V1, U2, V2, b1, b2 = [t.to(torch.float16) for t in [P, V1, U2, V2, b1, b2]]
    out = _impl(P, V1, U2, V2, b1, b2, BL=BL, BD=BD, BH=BH, BR1=BR1, BR2=BR2)
    return out.to(orig) if orig == torch.float32 else out
