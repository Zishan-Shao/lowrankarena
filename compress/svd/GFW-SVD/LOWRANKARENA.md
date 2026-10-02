# GFW-SVD Kronecker-factor generation in LowRankArena

[`../gfw_svd.py`](../gfw_svd.py) consumes precomputed Kronecker factors
(`--kron_factors_dir`). The [`kronfwsvd/`](./kronfwsvd/) directory holds the
code used to produce those factors for the LowRankArena LLaMA runs.

## Provenance

- Upstream repository:
  [`sayankotor/FisherKronecker`](https://github.com/sayankotor/FisherKronecker)
- Upstream commit for `kronfwsvd/`:
  [`d4e630aef4eb3ed4585270caeba15d236194f0c2`](https://github.com/sayankotor/FisherKronecker/commit/d4e630aef4eb3ed4585270caeba15d236194f0c2)
  (this differs from the `d009b02` commit pinned for `llama/`; see
  [`README.md`](./README.md))
- Patch relative to that upstream commit:
  [`LOWRANKARENA_UPSTREAM_DIFF.patch`](./LOWRANKARENA_UPSTREAM_DIFF.patch)

| File | Origin |
|---|---|
| [`kronfwsvd/krondecomp_cupy.py`](./kronfwsvd/krondecomp_cupy.py) | Upstream file, modified (see patch) |
| [`kronfwsvd/collect_weight_grads.py`](./kronfwsvd/collect_weight_grads.py) | LowRankArena, not in upstream |
| [`kronfwsvd/build_kron_layer.py`](./kronfwsvd/build_kron_layer.py) | LowRankArena, not in upstream |

## How the files fit together

Kronecker factors come out of two stages. `gfw_svd.py` then reads them.

- **`collect_weight_grads.py`** (gradient collection). It runs the model over
  a calibration set and registers gradient hooks on every projection module
  whose name matches `--layer_pattern`. For each effective batch
  (`batch_size × gradient_accumulation_steps`) it writes the accumulated
  per-weight gradients to a `grads_batch_*.safetensors` file.
- **`krondecomp_cupy.py`** (Kronecker decomposition). `get_kron_factors`
  takes the list of gradients for one layer and uses CuPy `svds` on the
  Fisher operator to find the Kronecker factors, returned as
  (output-side, input-side).
- **`build_kron_layer.py`** (factor files). It reads every
  `grads_batch_*.safetensors` in a gradient directory and calls
  `get_kron_factors` once per layer. For each layer it writes
  `<layer_name with . → _>.safetensors`, which contains:
  - `XF`: input-side factor (`in × in`)
  - `YF`: output-side factor (`out × out`)

  Upstream returns these factors in the opposite order. This script renames
  them to the `XF`/`YF` keys that `gfw_svd.py` expects.
- **[`../gfw_svd.py`](../gfw_svd.py)** (compression and export). It takes the
  factor directory through `kron_factors_dir`. A LLaMA model needs 7 factor
  files per layer (q/k/v/o/gate/up/down).

## Deliberate exclusions

Gradients, factor tensors, logs, and the run-orchestration shell scripts are
not part of the snapshot. Other files from the upstream `kronfwsvd/`, such as
`collect_grads.py`, `krondecomp_torch.py`, and the sanity checks, are also
excluded because this pipeline does not use them.
