# flashsvdffnv15_fp32.py — FP32-native v1.5 FFN entry point
#
# Difference from flashsvdffnv15.py:
#   - No fp32 → fp16 auto-cast.
#   - fp32 inputs run through the USE_FP32=1 kernel path:
#       * A100 / Ampere: TF32 tensor cores (fast, ~full dynamic range)
#       * Volta / Turing: CUDA fp32 FFMA cores (slower, exact fp32)
#   - fp16 / bf16 inputs unchanged (USE_FP32=0 path, same as v1.5).
#
# Use this wrapper when you want zero-cast overhead AND full fp32 range.
# Use flashsvdffnv15.py if you accept fp16 precision on FP32 input
# (slightly faster on non-Ampere GPUs due to smaller dtype).

import importlib.util
import os
import torch

_spec = importlib.util.spec_from_file_location(
    "_flashsvdffn_v15_fp32_impl",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "flashsvdffn", "flashsvdffn_v1.5_fp32.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_impl = _mod.flashsvd_ffn_fp32


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def flashsvd_ffn_v15_fp32(P, V1, U2, V2, b1, b2,
                           BL=64, BD=128, BH=64, BR1=32, BR2=32):
    """FP32-native v1.5 FFN wrapper.

    Accepts fp32, fp16, or bf16.  For fp32 inputs: uses USE_FP32=1 kernel
    path — no dtype cast, relies on TF32 tensor cores (A100) or CUDA FP32
    FFMA cores (Volta/Turing).

    BR2 is auto-set to next_pow2(R2) to satisfy Triton's tl.arange constraint.
    """
    # BR2 auto-alignment (power-of-2 constraint for Triton tl.arange)
    R2   = V2.shape[0]
    BR2  = _next_pow2(R2)
    return _impl(P, V1, U2, V2, b1, b2, BL=BL, BD=BD, BH=BH, BR1=BR1, BR2=BR2)
