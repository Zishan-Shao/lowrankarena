# LowRankArena

LowRankArena is the benchmark scaffold for low-rank and compression checkpoints. The normal path is: register a checkpoint, choose a benchmark lane, run the suite, and consume normalized JSON results. Re-running compression is optional and lives under `compress/`.

The repository has three paper-facing runtime surfaces:

- evaluation through standardized base and instruct lanes
- active GPU memory measurement through plain `transformers`
- serving and evaluation-speed measurement through `vLLM`

The current hosted checkpoint source of truth is the gated Hugging Face repository `Duke-CEI-SVD/LowRankArena`.

## Quick Map

- [`checkpoints/index.csv`](./checkpoints/index.csv): checkpoint registry used by all runners
- [`checkpoints/manifests/`](./checkpoints/manifests/README.md): optional rich checkpoint metadata
- [`benchmark/`](./benchmark/README.md): declarative suite definitions
- [`audit/`](./audit/README.md): appendix audit configs and reproducible job planning
- [`scripts/`](./scripts/README.md): user-facing entrypoints
- [`src/`](./src/README.md): reusable runtime, adapters, and result normalization
- [`results/`](./results/README.md): normalized benchmark outputs
- [`compress/`](./compress/README.md): optional artifact-generation layer

Benchmark YAML files describe the workload for one checkpoint. Checkpoint fan-out is controlled by registry selection, aggregate lane selection, or explicit `--checkpoint` overrides.

## Environment

Recommended environment on this machine:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate lowrankarena
python -V
```

To refresh an existing environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda env update -f environment.yml --prune
conda activate lowrankarena
python -V
```

Validated stack:

- Python `3.13.5`
- PyTorch `2.10.0+cu128`
- Transformers `4.57.6`
- vLLM `0.18.1`
- LM-Eval-Harness `0.4.11`

`vllm==0.18.1` requires Python `<3.14`, so Python `3.14+` can install the base runtime but will skip the vLLM speed dependency.

## Current Benchmark Lanes

There is intentionally no catch-all `benchmark/main.yaml`. Use explicit lanes for evaluation, and dedicated entries for system metrics.

```text
base = ppl + mcq + base/base_math
instruct = instruct/mmlu_pro + instruct/gsm8k
instruct_appendix = instruct/aime + instruct/ifeval
memory = memory/active
serving speed = speed/serve
online serving speed = speed/serve_e2e
evaluation speed = speed/speed
edge speed = speed/edge
```

Base/Universal evaluation:

- [`ppl.yaml`](./benchmark/ppl.yaml): WikiText-2 test PPL and C4 validation-stream PPL through the repo-owned contiguous runner. It uses raw text, `add_special_tokens=False`, non-overlapping blocks, and `max_length=2048`.
- [`mcq.yaml`](./benchmark/mcq.yaml): 0-shot BoolQ, HellaSwag, WinoGrande, PIQA, ARC-Easy, ARC-Challenge, and OpenBookQA through `lm-eval-harness`.
- [`base/base_math.yaml`](./benchmark/base/base_math.yaml): 0-shot base-safe math with local `lra_mathqa` plus upstream `mmlu_stem`, using MCQ/loglikelihood-style scoring. No CoT, no chat template, no free-form generation.

Instruct evaluation:

- [`instruct/mmlu_pro.yaml`](./benchmark/instruct/mmlu_pro.yaml): `leaderboard_mmlu_pro`, 5-shot direct-answer multiple choice, non-CoT, metric `acc`.
- [`instruct/gsm8k.yaml`](./benchmark/instruct/gsm8k.yaml): upstream `gsm8k_cot`, 8-shot CoT, greedy/default extraction, no self-consistency and no extra generation-length override in our YAML.
- [`instruct/aime.yaml`](./benchmark/instruct/aime.yaml): upstream `aime24`, 0-shot greedy generation, fixed exact-match extraction, and normalized solved count plus accuracy for appendix hard-math reporting.
- [`instruct/ifeval.yaml`](./benchmark/instruct/ifeval.yaml): upstream `ifeval`, chat-template protocol enabled, fixed upstream `max_gen_toks: 1280`, and strict/loose prompt-/instruction-level metrics.

System metrics:

- [`memory/active.yaml`](./benchmark/memory/active.yaml): active memory peak for one process via `torch.cuda.max_memory_allocated()`; normalized outputs record the CUDA device name/class and visible-device mapping.
- [`speed/serve.yaml`](./benchmark/speed/serve.yaml): mainstream vLLM serving cases for prefill/decode throughput; normalized outputs record the CUDA device name/class used by the run.
- [`speed/serve_e2e.yaml`](./benchmark/speed/serve_e2e.yaml): online serving benchmark that starts `vllm serve`, drives it with `vllm bench serve`, and records TTFT/TPOT/ITL/E2E latency plus request and output-token throughput.
- [`speed/speed.yaml`](./benchmark/speed/speed.yaml): end-to-end evaluation-speed route over the default base eval lane.
- [`speed/edge.yaml`](./benchmark/speed/edge.yaml): optional long-context serving cases for appendix or stress reporting.

## Backend Policy

The paper-facing `lm-eval-harness` suites default to `model_backend: vllm` so full runs do not crawl through the Transformers backend. The vLLM path reuses the project adapter in [`src/vllm/`](./src/vllm/README.md), including wrapper preparation for checkpoints that need it.

Use HF/Transformers only when you explicitly want a comparison or a local debug run:

```bash
python scripts/run_eval.py llama31-8b-instruct --suite instruct/mmlu_pro --model-backend hf --limit 1
python scripts/run_main.py --suites instruct --checkpoint llama31-8b-instruct --eval-model-backend hf --limit 1
```

PPL is the exception: `ppl.yaml` always uses the repo-owned contiguous PPL runner, even if an aggregate command passes a vLLM backend override.

Accuracy suites use `dtype: auto` by default so fp16 and bf16 checkpoints can run without editing YAML. Efficiency suites default to `float16` for paper-facing comparability; pass `--dtype bfloat16` when a fixed bf16 run is the intended comparison.

## Recommended Workflow

1. Select or register the checkpoint in [`checkpoints/index.csv`](./checkpoints/index.csv). For aggregate lanes, make sure its `variant` is `base` or `instruct`.
2. Run the right evaluation lane with [`scripts/run_main.py`](./scripts/run_main.py), or run one suite with [`scripts/run_eval.py`](./scripts/run_eval.py).
3. Run active memory separately with [`scripts/run_memory.py`](./scripts/run_memory.py).
4. Run serving or evaluation-speed separately with [`scripts/run_speed.py`](./scripts/run_speed.py).
5. Consume normalized outputs under [`results/`](./results/README.md). All `eval`, `memory`, and `speed` outputs share the project-owned schema from [`src/result_schema.py`](./src/result_schema.py).

Optional artifact-author path:

1. Build or plan an artifact through [`scripts/run_compress.py`](./scripts/run_compress.py).
2. Export it into a loadable checkpoint directory.
3. Register it in the checkpoint index or a sidecar manifest.
4. Use the normal evaluation, memory, and speed flow.

Appendix-audit path:

1. Pick an audit config under [`audit/configs/`](./audit/configs/).
2. Generate a command plan with [`audit/run_audit.py`](./audit/run_audit.py).
3. Run the generated script or the specialized priority-1 feasibility runner.
4. Summarize audit JSON into the appendix table/figure inputs.

## Common Commands

Base lane:

```bash
# run all enabled base checkpoints selected by benchmark/base.yaml
python scripts/run_main.py --suites base

# smoke one explicit base checkpoint
python scripts/run_main.py \
  --suites base \
  --checkpoint llama31-8b-svdllm-v1-update-0.6 \
  --limit 1 \
  --eval-tensor-parallel-size 1
```

Instruct lane:

```bash
# run all enabled instruct checkpoints selected by benchmark/instruct.yaml
python scripts/run_main.py --suites instruct

# run appendix-only instruct checks selected by benchmark/instruct_appendix.yaml
python scripts/run_main.py --suites instruct_appendix

# smoke one explicit instruct checkpoint
python scripts/run_main.py \
  --suites instruct \
  --checkpoint llama31-8b-instruct \
  --limit 1 \
  --eval-tensor-parallel-size 1
```

Single suites:

```bash
python scripts/run_eval.py llama31-8b-svdllm-v1-update-0.6 --suite mcq --limit 1
python scripts/run_eval.py llama31-8b-svdllm-v1-update-0.6 --suite base/base_math --limit 1
python scripts/run_eval.py llama31-8b-instruct --suite instruct/mmlu_pro --limit 1
python scripts/run_eval.py llama31-8b-instruct --suite instruct/gsm8k --limit 1
python scripts/run_eval.py llama31-8b-instruct --suite instruct/aime --limit 1
python scripts/run_eval.py llama31-8b-instruct --suite instruct/ifeval --limit 1
```

System metrics:

```bash
# active memory
python scripts/run_memory.py \
  llama-7b-svdllm-v1-update-0.5 \
  --suite memory/active \
  --batch-size 1 \
  --prompt-length 32 \
  --generation-length 8

# serving speed
python scripts/run_speed.py \
  llama-7b-svdllm-v1-update-0.5 \
  --suite speed/serve \
  --batch-size 1 \
  --prompt-length 32 \
  --generation-length 8 \
  --repeat 1 \
  --warmup 0

# evaluation speed over the default base eval route
python scripts/run_speed.py \
  llama-7b-svdllm-v1-update-0.5 \
  --suite speed/speed \
  --eval-limit 1

# online vLLM serving latency + throughput
python scripts/run_speed.py \
  llama-7b-svdllm-v1-update-0.5 \
  --suite speed/serve_e2e \
  --port 8000 \
  --num-prompts 8
```

Route smoke:

```bash
# demo.py checks load + eval + memory + serving-speed plumbing with tiny defaults.
# Its eval smoke intentionally uses the HF backend and --eval-limit=1.
CUDA_VISIBLE_DEVICES=0 python demo.py llama31-8b-svdllm-v1-update-0.6 --device cuda:0

# system-only smoke for a checkpoint family
CUDA_VISIBLE_DEVICES=0 python demo.py llama-7b-svdllm-v1-update-0.5 \
  --skip-eval \
  --device cuda:0 \
  --memory-prompt-length 16 \
  --memory-generation-length 4 \
  --speed-prompt-length 16 \
  --speed-generation-length 4 \
  --speed-repeat 1 \
  --speed-warmup 0 \
  --speed-gpu-memory-utilization 0.2
```

## Programmatic API

The intended public entrypoint is [`Arena`](./src/arena.py).

```python
from src import Arena

arena = Arena()

for row in arena.list(enabled_only=False):
    print(row["id"], row["method"], row["variant"])

print(arena.describe("llama31-8b-svdllm-v1-update-0.6")["subpath"])
```

`Arena` wraps the registry, loader, eval, memory, speed, and reporting modules without introducing a second object model.

## Notes

- `run_main.py` is evaluation-only and defaults to `--suites base`.
- Memory and speed use dedicated scripts because their runtime, metrics, and failure modes differ from accuracy evaluation.
- Materialized vLLM wrapper checkpoints belong in [`checkpoints/vllm/`](./checkpoints/vllm/README.md), not under `src/vllm/`.
- Benchmark outputs live under `results/`; generated smoke or leaderboard artifacts should not be committed unless intentionally curated.



# Environments

LowRankArena uses two repo-level environments:

- `lowrankarena`: benchmark loading, eval, memory, speed, and reporting. This is defined by [`../environment.yml`](../environment.yml).
- `compress`: compression, export, repair, and upload jobs. This is defined by [`compress.yml`](./compress.yml).

Create or update the compression environment with:

```bash
bash scripts/env/create_compress_env.sh
```

Then smoke-check it with:

```bash
conda run --no-capture-output -n compress \
  python scripts/env/check_compress_env.py --repo-root "$PWD"
```

Compression Slurm scripts default to `COMPRESS_CONDA_ENV=compress`. Evaluation Slurm scripts continue to activate `lowrankarena`; compression scripts that run a quick PPL smoke call `conda run -n ${LOWRANK_EVAL_CONDA_ENV:-lowrankarena}` for that evaluation step.

## Method Status

- SVD-LLM v1/v2: supported by `compress`.
- Basis Sharing: supported by `compress` for the current Qwen/GQA export path.
- MoDeGPT: supported by `compress`; the older `modegpt` env can stay as a fallback by setting `COMPRESS_CONDA_ENV=modegpt`.
- ASVD: compression/export imports under `compress`; the upstream direct eval path still expects the old `lm_eval.base` API, so use LowRankArena for benchmark evaluation.
- Dobi-SVD: source modules are restored for local compression/import checks. Its upstream dependency pins were older than the shared `compress` stack, so full GPU matrix reruns should still start with a smoke job before launching a large sweep.
