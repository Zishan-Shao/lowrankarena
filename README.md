# LowRankArena

This branch is an anonymous, benchmark-only artifact for double-blind review.
It keeps the benchmark definitions, runners, checkpoint registry schema, low-rank
runtime adapters, and result normalization code. It intentionally omits method
implementation code, generated benchmark outputs, paper figures, audit notes,
and non-anonymous checkpoint repository identifiers.

## What Is Included

- [`benchmark/`](./benchmark/README.md): declarative benchmark suite definitions
- [`checkpoints/index.csv`](./checkpoints/index.csv): seed checkpoint registry rows and placeholders
- [`checkpoints/manifests/`](./checkpoints/manifests/README.md): optional richer checkpoint metadata
- [`scripts/`](./scripts/README.md): command-line benchmark entrypoints
- [`src/`](./src/README.md): reusable runtime, adapters, and normalized output schema
- [`results/`](./results/README.md): empty output folders for regenerated benchmark artifacts
- [`tests/`](./tests/README.md): lightweight structural and runner-adjacent tests

The branch supports three benchmark surfaces:

- standardized accuracy/perplexity evaluation for base and instruction-tuned models
- active GPU memory measurement with `transformers`
- serving and evaluation-speed measurement with `vLLM`

## Environment

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

- Python `3.13.x`
- PyTorch `2.10.0`
- Transformers `4.57.6`
- vLLM `0.18.1`
- LM-Eval-Harness `0.4.11`

`vllm==0.18.1` requires Python `<3.14`, so Python `3.14+` can install the
base runtime but will skip the vLLM speed dependency.

## Benchmark Lanes

There is intentionally no catch-all `benchmark/main.yaml`. Use explicit lanes
for evaluation and dedicated entries for system metrics.

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

Base evaluation:

- [`ppl.yaml`](./benchmark/ppl.yaml): WikiText-2 test PPL and C4 validation-stream PPL through the contiguous runner.
- [`mcq.yaml`](./benchmark/mcq.yaml): 0-shot BoolQ, HellaSwag, WinoGrande, PIQA, ARC-Easy, ARC-Challenge, and OpenBookQA through `lm-eval-harness`.
- [`base/base_math.yaml`](./benchmark/base/base_math.yaml): base-safe math with 0-shot local `lra_mathqa` plus local 5-shot `MMLU_Math`.

Instruction-tuned evaluation:

- [`instruct/mmlu_pro.yaml`](./benchmark/instruct/mmlu_pro.yaml): `leaderboard_mmlu_pro`, 5-shot direct-answer multiple choice.
- [`instruct/gsm8k.yaml`](./benchmark/instruct/gsm8k.yaml): upstream `gsm8k_cot`, 8-shot CoT.
- [`instruct/aime.yaml`](./benchmark/instruct/aime.yaml): upstream `aime24`, 0-shot greedy generation.
- [`instruct/ifeval.yaml`](./benchmark/instruct/ifeval.yaml): upstream IFEval with chat-template protocol enabled.

System metrics:

- [`memory/active.yaml`](./benchmark/memory/active.yaml): active memory peak for one process via `torch.cuda.max_memory_allocated()`.
- [`speed/serve.yaml`](./benchmark/speed/serve.yaml): vLLM serving cases for prefill/decode throughput.
- [`speed/serve_e2e.yaml`](./benchmark/speed/serve_e2e.yaml): online serving benchmark using `vllm serve` plus `vllm bench serve`.
- [`speed/speed.yaml`](./benchmark/speed/speed.yaml): end-to-end evaluation-speed route over the base eval lane.
- [`speed/edge.yaml`](./benchmark/speed/edge.yaml): optional long-context serving stress cases.

## Checkpoints

The included registry contains public dense-model examples, a runnable anonymous
`lowrank-demo` row, and disabled low-rank checkpoint placeholders. Before
running other low-rank rows, update `checkpoints/index.csv` so each row points
to your anonymous Hugging Face model repository or to a local exported
checkpoint directory.

For local artifacts:

```bash
python scripts/add_checkpoint.py lowrank-demo \
  --source local \
  --model-family llama3.1 \
  --variant base \
  --method custom_lowrank \
  --subpath /path/to/exported/checkpoint \
  --benchmarks base speed \
  --notes "anonymous local artifact"
```

For anonymous hosted artifacts:

```bash
python scripts/add_checkpoint.py lowrank-demo \
  --source huggingface \
  --repo-id neurips-ed2026-anon-checkpoints/submission-checkpoints \
  --revision main \
  --model-family llama2 \
  --variant base \
  --method svdllm_v2 \
  --subpath full_checkpoint_zoo/llama2_7b/svdllm_v2/keep0p6 \
  --benchmarks base speed
```

## Common Commands

One-command anonymous PPL smoke check:

```bash
python scripts/run_eval.py lowrank-demo \
  --suite ppl_smoke \
  --device cpu \
  --batch-size 1 \
  --run-label smoke_ppl
```

This downloads an anonymous SVD-LLM checkpoint, runs a tiny contiguous PPL
route check, and writes normalized JSON under `results/eval/`. The first run
downloads the checkpoint shards; subsequent runs reuse the Hugging Face cache.
The `ppl_smoke` suite is intentionally tiny and should not be reported as a
benchmark metric.

The remaining examples assume the named checkpoint rows are present and
accessible. `lowrank-demo` is included for explicit smoke commands; full base
and speed runs need a GPU. The instruction-tuned examples use the public
`llama31-8b-instruct` row, which requires access to the upstream gated model, or
can be replaced with a local instruction-tuned row.

Base lane:

```bash
python scripts/run_main.py --suites base --limit 1

python scripts/run_main.py \
  --suites base \
  --checkpoint lowrank-demo \
  --limit 1 \
  --eval-tensor-parallel-size 1
```

Instruction-tuned lane:

```bash
python scripts/run_main.py --suites instruct --limit 1
python scripts/run_main.py --suites instruct_appendix --limit 1
```

Single suites:

```bash
python scripts/run_eval.py lowrank-demo --suite ppl_smoke --device cpu --batch-size 1
python scripts/run_eval.py lowrank-demo --suite mcq --limit 1
python scripts/run_eval.py lowrank-demo --suite base/base_math --limit 1
python scripts/run_eval.py llama31-8b-instruct --suite instruct/mmlu_pro --limit 1
python scripts/run_eval.py llama31-8b-instruct --suite instruct/gsm8k --limit 1
python scripts/run_eval.py llama31-8b-instruct --suite instruct/aime --limit 1
python scripts/run_eval.py llama31-8b-instruct --suite instruct/ifeval --limit 1
```

System metrics:

```bash
python scripts/run_memory.py lowrank-demo --suite memory/active

python scripts/run_speed.py lowrank-demo \
  --suite speed/serve \
  --batch-size 1 \
  --prompt-length 32 \
  --generation-length 8 \
  --repeat 1 \
  --warmup 0

python scripts/run_speed.py lowrank-demo \
  --suite speed/serve_e2e \
  --port 8000 \
  --num-prompts 8
```

Route smoke:

```bash
CUDA_VISIBLE_DEVICES=0 python demo.py lowrank-demo \
  --device cuda:0 \
  --memory-prompt-length 16 \
  --memory-generation-length 4 \
  --speed-prompt-length 16 \
  --speed-generation-length 4 \
  --speed-repeat 1 \
  --speed-warmup 0
```

## Programmatic API

```python
from src import Arena

arena = Arena()
for row in arena.list(enabled_only=False):
    print(row["id"], row["method"], row["variant"])

print(arena.describe("llama31-8b-dense")["subpath"])
```

## Notes For Anonymous Review

- Do not commit generated outputs that include absolute paths, usernames, tokens, or private repository IDs.
- Keep checkpoint artifacts in an anonymous repository or in a local path supplied outside the submission PDF.
- If this branch is mirrored from a non-anonymous GitHub owner, submit through an anonymity-preserving artifact service or an anonymous fork.
