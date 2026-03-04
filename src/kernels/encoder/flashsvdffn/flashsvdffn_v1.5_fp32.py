# flashsvdffn_v1.5_fp32.py
# FP32-native variant of the v1.5 rank-space FFN kernel.
#
# Differences from flashsvdffn_v1.5.py:
#   - Adds USE_FP32 constexpr: when True, in_dtype = tl.float32
#     → no fp16/bf16 cast, operands stay in fp32 throughout
#     → on Ampere (A100): tl.dot maps to TF32 tensor cores (fast, ~full range)
#     → on older GPUs:    tl.dot maps to CUDA fp32 FFMA cores (slower, exact)
#   - The TypeError for fp32 input is removed.
#   - fp16 / bf16 inputs still work identically to v1.5.
#
# Usage
# -----
#   from kernels.encoder_kernels.flashsvdffn.flashsvdffn_v1.5_fp32 import flashsvd_ffn_fp32
#   out = flashsvd_ffn_fp32(P_fp32, V1, U2, V2, b1, b2)  # native fp32, no cast

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_ffn_fp32(
    P_ptr, V1_ptr, U2_ptr, V2_ptr, C_ptr,
    b1_ptr, b2_ptr,
    B, L, D, H, R1, R2,
    sP_b, sP_l, sP_r1,
    sV1_r1, sV1_d,
    sU2_d, sU2_r2, sV2_r2, sV2_h,
    sb1, sb2,
    sC_b, sC_l, sC_h,
    BL: tl.constexpr, BD: tl.constexpr, BH: tl.constexpr,
    BR1: tl.constexpr, BR2: tl.constexpr,
    R2_PAD: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32: tl.constexpr,   # NEW: native fp32 path
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)

    offs_Pb = pid_b * sP_b
    offs_Cb = pid_b * sC_b
    offs_l  = pid_l * BL + tl.arange(0, BL)

    # Select in_dtype based on constexpr flags
    if USE_FP32:
        in_dtype = tl.float32      # native fp32; TF32 tensor core on A100
    elif USE_BF16:
        in_dtype = tl.bfloat16
    else:
        in_dtype = tl.float16

    offs_r2 = tl.arange(0, R2_PAD)
    m_r2    = offs_r2 < R2
    acc_r   = tl.zeros((BL, R2_PAD), dtype=tl.float32)
    T_acc   = tl.zeros((BL, BD),     dtype=tl.float32)

    for d0 in range(0, D, BD):
        d   = d0 + tl.arange(0, BD)
        m_d = d < D

        T_acc *= 0.0
        for r1_0 in range(0, R1, BR1):
            r1   = r1_0 + tl.arange(0, BR1)
            m_r1 = r1 < R1
            P_sub = tl.load(
                P_ptr + offs_Pb
                      + offs_l[:, None] * sP_l
                      + r1[None, :] * sP_r1,
                mask=(offs_l[:, None] < L) & m_r1[None, :],
                other=0.0,
            )
            V1_sub = tl.load(
                V1_ptr + r1[:, None] * sV1_r1 + d[None, :] * sV1_d,
                mask=m_r1[:, None] & m_d[None, :],
                other=0.0,
            )
            T_acc += tl.dot(P_sub, V1_sub)
        b1_sl  = tl.load(b1_ptr + d * sb1, mask=m_d, other=0.0)
        T_acc += b1_sl[None, :]

        # Exact GELU (erf)
        T = T_acc * 0.5 * (1.0 + tl.erf(T_acc / tl.sqrt(2.0)))
        T = T.to(in_dtype)

        U2_sub = tl.load(
            U2_ptr + d[:, None] * sU2_d + offs_r2[None, :] * sU2_r2,
            mask=m_d[:, None] & m_r2[None, :],
            other=0.0,
        ).to(in_dtype)
        acc_r = tl.dot(T, U2_sub, acc=acc_r, out_dtype=tl.float32)

    # single lift: S @ V2 + b2
    acc_r_f = acc_r.to(in_dtype)
    m_l  = offs_l < L
    base = C_ptr + offs_Cb + offs_l[:, None] * sC_l
    for h0 in range(0, H, BH):
        offs_h = h0 + tl.arange(0, BH)
        V2_sub = tl.load(
            V2_ptr + offs_r2[:, None] * sV2_r2 + offs_h[None, :] * sV2_h,
            mask=m_r2[:, None] & (offs_h[None, :] < H),
            other=0.0,
        ).to(in_dtype)
        acc = tl.dot(acc_r_f, V2_sub, out_dtype=tl.float32)
        b2_sl = tl.load(b2_ptr + offs_h * sb2, mask=(offs_h < H), other=0.0)
        acc  += b2_sl[None, :]
        mask  = m_l[:, None] & (offs_h[None, :] < H)
        tl.store(base + offs_h[None, :] * sC_h, acc, mask=mask)


def flashsvd_ffn_fp32(
    P, V1, U2, V2, b1, b2,
    BL=64, BD=128, BH=64,
    BR1=32, BR2=32,
    *,
    num_warps: int = 4,
    num_stages: int = 2,
):
    """
    FP32-native v1.5 FFN kernel.

    Accepts float32, float16, or bfloat16 inputs.
    For fp32: uses tl.float32 in_dtype — no cast, uses TF32 tensor cores on A100
              or CUDA fp32 FFMA on older GPUs (full precision, no precision loss
              from fp16 cast).
    """
    B, L, R1 = P.shape
    _, D      = V1.shape
    _, H      = V2.shape
    R2        = U2.shape[1]

    # ── auto-align BR2 to next_pow2(R2) so R2_PAD is a power of 2 ──────
    def _nxt2(n): return 1 if n <= 1 else 1 << (n - 1).bit_length()
    BR2   = _nxt2(R2)
    R2_PAD = ((R2 + BR2 - 1) // BR2) * BR2

    for name, val in (("BL", BL), ("BD", BD), ("BH", BH), ("BR1", BR1)):
        if int(val) < 16:
            raise ValueError(f"{name} must be >= 16 for tl.dot (got {val}).")

    is_fp32 = (P.dtype == torch.float32)
    is_bf16 = (P.dtype == torch.bfloat16)

    C = torch.empty((B, L, H), device=P.device, dtype=P.dtype)
    strides = dict(
        sP_b=P.stride(0), sP_l=P.stride(1), sP_r1=P.stride(2),
        sV1_r1=V1.stride(0), sV1_d=V1.stride(1),
        sU2_d=U2.stride(0), sU2_r2=U2.stride(1),
        sV2_r2=V2.stride(0), sV2_h=V2.stride(1),
        sb1=b1.stride(0), sb2=b2.stride(0),
        sC_b=C.stride(0), sC_l=C.stride(1), sC_h=C.stride(2),
    )
    grid = (B, triton.cdiv(L, BL))
    _fused_ffn_fp32[grid](
        P, V1, U2, V2, C, b1, b2,
        B, L, D, H, R1, R2,
        *strides.values(),
        BL, BD, BH, BR1, BR2,
        R2_PAD=R2_PAD,
        USE_BF16=int(is_bf16),
        USE_FP32=int(is_fp32),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return C
