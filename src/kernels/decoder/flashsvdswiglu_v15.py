"""Re-export flashsvd_ffn_swiglu from flashsvd-v1.5 decode-opt kernel.

Uses importlib because the filename contains dots (v1.5).
Falls back to the older src/kernels/decoder/flashsvdswiglu.py if v1.5 not found.
"""
import os, importlib.util

_REPO = os.path.dirname(          # lowrankarena/
    os.path.dirname(               # src/
        os.path.dirname(           # kernels/
            os.path.dirname(os.path.abspath(__file__))  # decoder/
        )
    )
)

_V15_PATH = os.path.join(
    _REPO, "flashsvd-v1.5", "flashsvd-v1.5",
    "flashsvdswiluffn", "flashsvdswiglu_v1.5_decode_opt.py",
)

if os.path.isfile(_V15_PATH):
    _spec = importlib.util.spec_from_file_location("_flashsvdswiglu_decode_opt", _V15_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    flashsvd_ffn_swiglu = _mod.flashsvd_ffn_swiglu
else:
    from src.kernels.decoder.flashsvdswiglu import flashsvd_ffn_swiglu  # noqa: F401

__all__ = ["flashsvd_ffn_swiglu"]
