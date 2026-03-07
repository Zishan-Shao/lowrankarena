"""Re-export token-level reconstruct functions from flashsvd-v1.5 decode kernel."""
import os, importlib.util

_REPO = os.path.dirname(          # lowrankarena/
    os.path.dirname(               # src/
        os.path.dirname(           # kernels/
            os.path.dirname(os.path.abspath(__file__))  # decoder/
        )
    )
)

_PATH = os.path.join(
    _REPO, "flashsvd-v1.5", "flashsvd-v1.5",
    "flashsvdropeattn_short", "flashsvdropeattn_dense_decode.py",
)

_spec = importlib.util.spec_from_file_location("_flashsvdropeattn_dense_decode", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

reconstruct_qkv_token        = _mod.reconstruct_qkv_token         # noqa: F401
reconstruct_qkv_token_shared = _mod.reconstruct_qkv_token_shared   # noqa: F401

__all__ = ["reconstruct_qkv_token", "reconstruct_qkv_token_shared"]
