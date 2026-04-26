# `compress/`

`compress/` is the optional artifact-generation layer for LowRankArena.

## Intended Use

LowRankArena has two clean paths:

1. Default reviewer path:
   - load released checkpoints
   - run unified eval
   - run unified memory measurement
   - run unified speed tests
   - export tables
2. Optional author/extender path:
   - start from a dense model
   - generate a compressed artifact
   - save it in a benchmark-friendly format
   - optionally register it into `checkpoints/index.csv`
   - then run the same eval, memory, and speed flow

The main benchmark does **not** depend on re-running compression. `compress/` exists for transparency, small-scale reruns, and extensions to new methods.

## Family Semantics

`compress/` groups methods by artifact-generation family, but the families are not all expected to behave the same way operationally.

### `compress/svd/`

`compress/svd/` is the home for low-rank methods such as:

- ASVD
- SVD-LLM
- Dobi-SVD
- basis sharing
- plain SVD
- FWSVD

These methods often need:

- their own calibration recipe
- method-specific recovery or remapping logic
- method-specific export code
- sometimes their own dependency stack

So `compress/svd/` should be read as:

> a uniform LowRankArena wrapper layer around heterogeneous low-rank builders

not as:

> one monolithic runtime that guarantees every SVD baseline can be imported into the same Python process

### `compress/prune/`

`compress/prune/` is the corresponding layer for structured pruning baselines such as:

- SliceGPT
- LLM-Pruner
- Bonsai
- Wanda-SP

This family is intentionally parallel to `compress/svd/`:

- each method gets a thin `build(request)` entrypoint
- each method is free to call an external baseline repo or a local implementation
- each method writes the same artifact metadata shape

The important unification point is the artifact contract, not the internal pruning runtime.

### `compress/quant/`

`compress/quant/` stays in the tree because quantization is the most realistic family to support locally in a shared environment.

In practice, quant baselines are more likely to be runnable with:

- the main `lowrankarena` environment
- modern `transformers`
- local GPU kernels such as FlashAttention-2 and vLLM-adjacent stacks

So quantization remains a first-class local path even if some pruning and SVD baselines eventually use separate plugin environments.

## Design Rules

- `compress/` only builds artifacts. It does not run eval, speed, or reporting.
- Generated artifacts write a uniform `manifest.json`.
- If an artifact becomes loadable as a local Hugging Face-style checkpoint, it can be registered into `checkpoints/index.csv`.
- Third-party baseline code is treated as optional support code, not as the main benchmark framework.
- The unification target is the exported artifact format, not a single universal dependency environment.

## Environment Strategy

LowRankArena keeps benchmark evaluation and artifact generation in separate environments:

- `lowrankarena`: loading, eval, memory, speed, and reporting.
- `compress`: compression, export, repair, and upload jobs.

The compression environment is defined in [`../envs/compress.yml`](../envs/compress.yml) and can be created with:

```bash
bash scripts/env/create_compress_env.sh
```

Compression Slurm scripts default to `COMPRESS_CONDA_ENV=compress`. Evaluation Slurm scripts keep using `lowrankarena`; mixed compression scripts that run a quick PPL smoke invoke `LOWRANK_EVAL_CONDA_ENV=lowrankarena` for that step.

We still do **not** assume that every historical baseline can share one perfect runtime. The `compress` environment is the default supported runtime for the current SVD-LLM, Basis Sharing, and MoDeGPT artifact paths. Older upstream baselines may still need method-specific repair:

- ASVD compression/export imports under `compress`; its upstream direct eval path still expects the legacy `lm_eval.base` API.
- Dobi-SVD's restored source imports under `compress`, but its upstream pins were older than the shared stack, so large GPU matrix reruns should start with a smoke job.

The unification target remains the exported artifact format, with `lowrankarena` as the stable benchmark runtime.

## Current Layout

```text
compress/
├── README.md
├── common.py
├── save.py
├── artifacts/
├── svd/
│   ├── asvd.py
│   ├── basis_sharing.py
│   ├── dobi_svd.py
│   ├── fwsvd.py
│   ├── svd.py
│   └── svd_llm.py
├── prune/
│   ├── bonsai.py
│   ├── llm_pruner.py
│   ├── slicegpt.py
│   └── wanda_sp.py
└── quant/
    ├── awq.py
    ├── gptq.py
    └── rtn.py
```

## Baselines

Some SVD baselines are already vendored locally:

- `compress/svd/ASVD`
- `compress/svd/Dobi-SVD`
- `compress/svd/SVD-LLM`
- `compress/svd/Basis_Sharing`

Other baselines are resolved through the registry in `compress/common.py` and can be cloned on demand by `scripts/run_compress.py --clone-baseline`.

This is deliberate. For `svd/` and `prune/`, LowRankArena should prefer:

- vendored snapshots when the project already depends on them
- otherwise an explicit baseline registry plus on-demand clone

instead of pretending every method already belongs to one local framework.

## Example

```bash
python scripts/run_compress.py \
  --family svd \
  --method asvd \
  --model meta-llama/Llama-3.1-8B \
  --ratio 0.5
```

The current scaffold writes:

- `manifest.json`
- `compression_log.json`
- `planned_command.sh` when a command template is known

This is enough to make the artifact flow explicit before wiring in each baseline end-to-end.

## Reviewer Guidance

Reviewers should usually ignore `compress/` unless they specifically want to inspect artifact generation.

The recommended interpretation is:

- `src/` + `scripts/run_eval.py` + `scripts/run_speed.py` are the main benchmark path
- `compress/svd/` and `compress/prune/` are extension and transparency layers
- `compress/quant/` is kept as a practical local path because quant methods are more feasible to run in a shared local stack
