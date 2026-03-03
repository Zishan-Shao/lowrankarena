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


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n (required by tl.arange in Triton kernels)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def flashsvd_ffn_v15(P, V1, U2, V2, b1, b2, BL=64, BD=128, BH=64, BR1=32, BR2=32):
    """v1.5 FFN with fp32 auto-cast fallback.

    When the model runs in fp16/bf16, tensors arrive at native dtype → zero
    cast overhead.  When running fp32 (e.g. --dtype fp32), we cast to fp16
    for the kernel and restore dtype on output.

    BR2 is auto-set to next_pow2(R2) so that R2_PAD = ceil(R2/BR2)*BR2 is
    always a power of 2 (Triton tl.arange constraint).
    For rank_ff>256 (BR2=512): BL is capped at 32 and num_stages=1 to
    keep Required=1×(32×512×4+32×128×2)=73728 bytes within the hardware
    shared memory limit (~101376 bytes).
    """
    orig = P.dtype
    if orig == torch.float32:
        # fallback: cast to fp16 for kernel, restore dtype on output
        P, V1, U2, V2, b1, b2 = [t.to(torch.float16) for t in [P, V1, U2, V2, b1, b2]]
    # Triton requires tl.arange(0, R2_PAD) to be power of 2.
    # R2_PAD = ceil(R2 / BR2) * BR2; setting BR2 = next_pow2(R2) gives
    # R2_PAD = next_pow2(R2) which is always a power of 2.
    R2 = V2.shape[0]
    BR2 = _next_pow2(R2)
    # Required = num_stages × (BL×R2_PAD×4 + BL×BD×2).
    # acc_r=[BL,R2_PAD] fp32 spills to shared memory and dominates.
    # For BR2>=512 (rank_ff>256, e.g. adasvd): reduce BL and num_stages
    # to stay within hardware shared memory limit (~101376 bytes).
    # num_stages=1, BL=32, BD=128, BR2=512 → 1×(65536+8192)=73728 ✓
    ns = 2
    if BR2 >= 512:
        BL = min(BL, 32)
        ns = 1
    out = _impl(P, V1, U2, V2, b1, b2, BL=BL, BD=BD, BH=BH, BR1=BR1, BR2=BR2,
                num_stages=ns)
    return out.to(orig) if orig == torch.float32 else out
