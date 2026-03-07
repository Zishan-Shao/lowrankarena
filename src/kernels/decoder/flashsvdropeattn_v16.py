"""Re-export flashsvd v1.6 attention API (prefill + decode with low-rank KV cache).

Prefill:
  flashsvd_attn_packed(f: PackedFactors, rotary_cos, rotary_sin, ...)
  flashsvd_attn_varlen(f: VarlenFactors, rotary_cos, rotary_sin, ...)

Decode (low-rank KV cache, split-K):
  flashsvd_attn_decode_packed(f: DecodePackedFactors, rotary_cos, rotary_sin, seqlen_k=...)

Helpers:
  build_rope_tables(seqlen, head_dim, base, device, dtype) -> (cos [S, dh/2], sin [S, dh/2])
  PackedFactors, VarlenFactors, DecodePackedFactors, DecodeVarlenFactors
"""
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
    "flashsvdropeattn_short", "flashsvdropeattn_v1.6_decode_opt.py",
)

_spec = importlib.util.spec_from_file_location("_flashsvdropeattn_v16", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Data containers
PackedFactors        = _mod.PackedFactors         # noqa: F401
VarlenFactors        = _mod.VarlenFactors          # noqa: F401
DecodePackedFactors  = _mod.DecodePackedFactors    # noqa: F401
DecodeVarlenFactors  = _mod.DecodeVarlenFactors    # noqa: F401

# Helpers
build_rope_tables    = _mod.build_rope_tables      # noqa: F401

# Prefill kernels
flashsvd_attn_packed  = _mod.flashsvd_attn_packed   # noqa: F401
flashsvd_attn_varlen  = _mod.flashsvd_attn_varlen   # noqa: F401

# Decode kernels (v2 — last binding in the module overrides v1)
flashsvd_attn_decode_packed  = _mod.flashsvd_attn_decode_packed   # noqa: F401
flashsvd_attn_decode_varlen  = _mod.flashsvd_attn_decode_varlen   # noqa: F401

__all__ = [
    "PackedFactors", "VarlenFactors", "DecodePackedFactors", "DecodeVarlenFactors",
    "build_rope_tables",
    "flashsvd_attn_packed", "flashsvd_attn_varlen",
    "flashsvd_attn_decode_packed", "flashsvd_attn_decode_varlen",
]
