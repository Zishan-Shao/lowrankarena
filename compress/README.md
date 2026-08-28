# `compress/`

`compress/` is the optional artifact-generation layer for LowRankArena.

## Unified CLI contract

Install the repository-owned benchmark and compression adapter stack once:

```bash
python -m pip install -r compress/requirements.txt
```

Then inspect method capability before allocating GPUs or downloading weights:

```bash
python scripts/run_compress.py --list-methods
python scripts/run_compress.py \
  --family svd \
  --method gfw_svd \
  --model meta-llama/Llama-3.1-8B \
  --ratio 0.5 \
  --extra kron_factors_dir=/path/to/factors \
  --preflight-only
```

The default command writes an auditable plan. `--execute` is accepted only for
methods whose LowRankArena adapter has an end-to-end exporter. Unsupported
execution is rejected during preflight instead of silently writing a plan or
starting an upstream script that cannot produce a loadable artifact.

The unified CLI is a thin wrapper, not a replacement for the recorded
reproduction commands. Existing commands in
`scripts/run_aasvd_keep_sweep_20260724.sh` and
`scripts/run_new_method_keep_sweep_20260724.sh` remain valid and define the
default numerical recipe. In particular, the wrappers retain AA-SVD's bf16
compression, fp16 HF export, Swift-SVD's fp16 export, and ZS-SVD seed 3.

Current end-to-end adapters are AA-SVD, GFW-SVD, Swift-SVD, and ZS-SVD. Every
other public method has an importable planning adapter, but remains explicitly
execution-gated until its native output is connected to the shared artifact
contract.

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

`compress/quant/` keeps the planned AWQ, GPTQ, and RTN integration points in
the same registry. None of those three wrappers is currently advertised as
end-to-end executable: their exporters and runtime-specific loading contracts
must be validated before `--execute` is enabled.

## Design Rules

- `compress/` only builds artifacts. It does not run eval, speed, or reporting.
- Generated artifacts write a uniform `manifest.json`.
- If an artifact becomes loadable as a local Hugging Face-style checkpoint, it can be registered into `checkpoints/index.csv`.
- Third-party baseline code is treated as optional support code, not as the main benchmark framework.
- The unification target is the exported artifact format, not a single universal dependency environment.

## Environment Strategy

LowRankArena-owned adapters and exporters use the additive pinned stack in
[`compress/requirements.txt`](./requirements.txt), which inherits the root
benchmark requirements without changing them. Historical requirement files
inside vendored upstream snapshots are provenance records and must not be
installed over that stack.

Current expectation:

- `svd/` methods may need per-method or per-baseline environments because many upstream repos pin incompatible `torch`, `transformers`, `lm_eval`, or `flash-attn` versions.
- `prune/` methods may also need separate environments for the same reason, especially older research repos.
- `quant/` methods are the most likely to run directly inside the main `lowrankarena` environment.

Some unadapted upstream research entrypoints still document mutually
incompatible historical environments. They are kept for source audit, while
the unified CLI refuses end-to-end execution until compatibility work and an
HF export path have both been validated. This means LowRankArena converges
toward:

- one stable benchmark environment for loading, eval, speed, and reporting
- optional method-specific compression environments for artifact generation

In other words:

> shared benchmark runtime, conditional compression runtimes

That separation keeps the main benchmark path simple while still making compression-time code transparent.

## Current Layout

```text
compress/
├── README.md
├── common.py
├── save.py
├── artifacts/
├── svd/
│   ├── asvd.py
│   ├── aa_svd.py
│   ├── basis_sharing.py
│   ├── dobi_svd.py
│   ├── fwsvd.py
│   ├── gfw_svd.py
│   ├── svd.py
│   ├── svd_llm.py
│   ├── swift_svd.py
│   └── zs_svd.py
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

## Planning-only example

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

This records the intended artifact flow, but does not claim that ASVD is
end-to-end executable. Check `supports_execute` in `--list-methods` before
adding `--execute`.

## Reviewer Guidance

Reviewers should usually ignore `compress/` unless they specifically want to inspect artifact generation.

The recommended interpretation is:

- `src/` + `scripts/run_eval.py` + `scripts/run_speed.py` are the main benchmark path
- `compress/svd/`, `compress/prune/`, and `compress/quant/` are extension and
  transparency layers whose execution capability is reported method by method
