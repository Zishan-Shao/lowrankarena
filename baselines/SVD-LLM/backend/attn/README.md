# FlashSVD Attention Backends

This folder holds backend-facing attention decode helpers.

It exists for symmetry with `backend/mlp/`, but it is not a free-form runtime
backend registry. The component still owns attention semantics in
`flashsvd_component/svd_llama.py`.

## What is active

- Current dense-token decode helper loading:
  - [`../../kernels/flashsvdropeattn_dense_decode.py`](../../kernels/flashsvdropeattn_dense_decode.py)
- Archived low-rank decode module loading:
  - [`../../kernels/flashsvd-archive/v1.5/flashsvdropeattn/`](../../kernels/flashsvd-archive/v1.5/flashsvdropeattn/)
  - [`../../kernels/flashsvd-archive/v1.6/flashsvdropeattn/`](../../kernels/flashsvd-archive/v1.6/flashsvdropeattn/)
- Main component dispatch:
  - [`../../flashsvd_component/svd_llama.py`](../../flashsvd_component/svd_llama.py)

## Ownership

- `backend/attn/`
  runtime helper code and legacy decode-module loading
- `kernels/`
  current helper kernels and archived versioned implementations
- `benchmark/`
  attention comparisons and regressions
- `flashsvd_component/legacy/`
  compatibility re-exports and low-rank cache state only
