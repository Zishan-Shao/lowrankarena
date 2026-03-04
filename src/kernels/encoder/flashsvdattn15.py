# flashsvdattn15.py — v1.5 attention entry point (rank-space kernel)
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "_flashsvdattn_v15_impl",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "flashsvdattn", "flashsvdattn_v1.5.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
flash_svd_attention_v15 = _mod.flash_svd_attention
