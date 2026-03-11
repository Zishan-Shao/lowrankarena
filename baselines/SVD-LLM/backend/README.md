# FlashSVD-v1.5 Backends

This folder holds backend-specific runtime implementations.

Rules:

- component semantics stay in `flashsvd_component/`
- low-level historical kernel archives stay in `kernels/flashsvd-archive/`, split by version under `v1/`, `v1.5/`, and `v1.6/`
- benchmark and compare code stay in `benchmark/`
- backend-specific production or experimental entrypoints live here

Current layout:

- `backend/attn/`
  attention decode helper loading and legacy decode-module resolution
- `backend/mlp/`
  MLP exact decode implementations and explicit experimental backends
