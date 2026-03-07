"""Re-export flashsvd_ffn_swiglu from decoder kernel (v1.5 decode-opt)."""
from src.kernels.decoder.flashsvdswiglu_v15 import flashsvd_ffn_swiglu  # noqa: F401

__all__ = ["flashsvd_ffn_swiglu"]
