# `benchmark/`

This directory defines the public benchmark surface of LowRankArena.

It contains declarative suite specifications only. Execution logic lives in [`scripts/`](../scripts/README.md) and reusable runtime code lives in [`src/`](../src/README.md).

## Structure

- [`base.yaml`](./base.yaml): paper-facing Base/Universal evaluation lane. It runs shared retention suites plus base-only tasks from [`base/`](./base/README.md).
- [`instruct.yaml`](./instruct.yaml): instruct-model main leaderboard lane for [`instruct/`](./instruct/README.md) tasks.
- [`instruct_appendix.yaml`](./instruct_appendix.yaml): instruct-only appendix/sanity lane for hard math and instruction-following stress tests.
- [`ppl.yaml`](./ppl.yaml): shared contiguous-block perplexity suite over `WikiText-2` test and a fixed-budget `C4` validation stream.
- [`ppl_smoke.yaml`](./ppl_smoke.yaml): tiny contiguous-block PPL smoke suite for checking that an artifact can download, load, tokenize, score, and write normalized outputs.
- [`mcq.yaml`](./mcq.yaml): shared multi-task multiple-choice suite for headline commonsense QA reporting.
- [`mmlu.yaml`](./mmlu.yaml): shared dedicated MMLU suite using the official `mmlu` group.
- [`base/`](./base/README.md): base-only suites that should not depend on chat formatting or long-form generation behavior.
- [`instruct/`](./instruct/README.md): instruct-only suites for direct-answer MMLU-Pro, GSM8K CoT generation, AIME, and IFEval.
- [`memory/active.yaml`](./memory/active.yaml): dedicated active GPU memory workload parameters for one checkpoint.
- [`speed/`](./speed/README.md): dedicated speed suites backed by vLLM plus an evaluation-speed suite that measures end-to-end benchmark runtime.

The YAML files define per-checkpoint workloads and suite composition. Checkpoint fan-out is controlled by runner selection, registry metadata, or explicit `--checkpoint` overrides.

## Evaluation Lanes

- Shared suites at the root of `benchmark/` can be run on either base or instruct checkpoints when explicitly requested.
- Base-only suites belong under `benchmark/base/`.
- Instruct-only suites belong under `benchmark/instruct/`.
- Aggregate entries control model-lane selection by registry metadata. `base.yaml` selects checkpoints whose `variant` is `base`, while `instruct.yaml` selects records whose `variant` is `instruct`. Suites do not hard-code checkpoint names.

The current base lane is intentionally conservative:

```text
base = ppl + mcq + base/base_math
```

It measures language-model retention, multiple-choice accuracy, and base-safe math without relying on chat templates or long Chain-of-Thought generation. `base/base_math` uses a local script-free 0-shot `lra_mathqa` wrapper plus a local 5-shot `MMLU_Math` MCQ group.

The current instruct-related lanes are:

```text
instruct = instruct/mmlu_pro + instruct/gsm8k
instruct_appendix = instruct/aime + instruct/ifeval
```

It contains only the Instruct leaderboard tasks: direct-answer MMLU-Pro plus GSM8K Chain-of-Thought generation for chat/instruction-tuned checkpoints. Shared retention/MCQ probes can still be run explicitly, but they are not part of the Instruct table entrypoint.

The appendix lane keeps slower or more protocol-sensitive instruct-only checks separate from the main table. `instruct/aime` uses `aime24` with upstream exact-match extraction and records both accuracy and solved count. `instruct/ifeval` uses upstream IFEval metrics, applies the tokenizer chat template consistently across models, and reports the strict prompt-level metric as the headline while preserving strict/loose prompt- and instruction-level metrics.

There is intentionally no catch-all `main.yaml`. Paper accuracy/perplexity tables should use the explicit `base` or `instruct` lanes, while system metrics use their own dedicated entries:

```text
memory = memory/active
serving speed = speed/serve
online serving speed = speed/serve_e2e
evaluation speed = speed/speed
edge speed = speed/edge
```

## Reproducibility

Suite membership and task selection are version-controlled in YAML. For `lm_eval_harness` suites, the concrete task IDs live under each suite's `eval.tasks`.

Accuracy suites use `dtype: auto` by default so fp16 and bf16 checkpoints can run without editing YAML. Efficiency suites default to `float16` for comparable system measurements; override from the CLI when a run needs fixed bf16, for example `--dtype bf16` on `scripts/run_memory.py` or `scripts/run_speed.py`, or `--eval-dtype bf16` on `scripts/run_main.py`.

The paper-facing `lm_eval_harness` suites for base MCQ, base math, MMLU-Pro, GSM8K, AIME, and IFEval default to `model_backend: vllm` so full leaderboard runs do not crawl through the Transformers backend. Use `--model-backend hf` on `scripts/run_eval.py` or `--eval-model-backend hf` on `scripts/run_main.py` when you need a direct HF comparison. `ppl.yaml` stays on the project-owned contiguous PPL runner and ignores vLLM backend overrides.

`ppl.yaml` is different because it uses the project-owned `contiguous_ppl` backend. Its concrete evaluation text is defined under `eval.datasets`: `WikiText-2` uses a pinned dataset revision plus the full `test` split, while `c4_stream` uses a pinned `allenai/c4` revision, `validation` split, `config_name: en`, `document_offset: 0`, and a fixed token budget. The runner records the scored token SHA-256 hash for each dataset so repeated runs can verify they evaluated the same token sequence.

Efficiency outputs record the CUDA runtime metadata under `runtime.cuda_runtime`, including each visible/selected device's name, compute capability, and total memory. Compare memory and speed numbers only within the same GPU class.

## Placement Rules

- Put suites that are valid for both pretrained/base and instruction-tuned checkpoints at the root of `benchmark/`.
- Put suites that assume no chat template, no instruction-following behavior, or only loglikelihood/short-answer scoring under `benchmark/base/`.
- Put suites that target instruction-tuned checkpoints, direct-answer instruction prompting, free-form answer extraction, or full-generation reasoning under `benchmark/instruct/`.
- Do not add `instruct/` suites to the base lane unless the benchmark protocol is rewritten to be base-safe.
- If a future MMLU-Pro base-safe wrapper is added, it belongs under `benchmark/base/`; the existing `instruct/mmlu_pro.yaml` is the instruct-leaderboard direct-answer protocol.

## Running

The top-level README includes a CPU-only anonymous PPL smoke command that runs
directly from this branch. The examples below assume the named checkpoint rows
are present and accessible; `lowrank-demo` is included for explicit smoke
commands, while full base and speed runs need a GPU.

```bash
# One shared suite
python scripts/run_eval.py lowrank-demo --suite ppl_smoke --device cpu --batch-size 1

# Base lane
python scripts/run_main.py --suites base --limit 1

# Instruct lane
python scripts/run_main.py --suites instruct --limit 1

# Instruct appendix lane
python scripts/run_main.py --suites instruct_appendix --limit 1

# Explicit checkpoint override
python scripts/run_main.py --suites base --checkpoint lowrank-demo --limit 1

# Dedicated system metrics
python scripts/run_memory.py lowrank-demo --suite memory/active
python scripts/run_speed.py lowrank-demo --suite speed/serve
python scripts/run_speed.py lowrank-demo --suite speed/serve_e2e

# Single lane-specific suites
python scripts/run_eval.py lowrank-demo --suite base/base_math --limit 1
python scripts/run_eval.py llama31-8b-instruct --suite instruct/mmlu_pro --limit 1
python scripts/run_eval.py llama31-8b-instruct --suite instruct/gsm8k --limit 1
python scripts/run_eval.py llama31-8b-instruct --suite instruct/aime --limit 1
python scripts/run_eval.py llama31-8b-instruct --suite instruct/ifeval --limit 1

# HF fallback for comparison/debugging
python scripts/run_eval.py llama31-8b-instruct --suite instruct/mmlu_pro --model-backend hf --limit 1
```

Aggregate suites pass their `selection` down to included shared suites. This lets `base.yaml` and `instruct.yaml` reuse root-level `ppl` and `mcq` while still selecting the right checkpoint lane. Passing `--checkpoint` to `scripts/run_main.py` overrides YAML selection and runs the requested suite set for that explicit checkpoint.

## Design Intent

- Keep benchmark definitions human-readable and reviewable.
- Make suite membership and task selection explicit in version-controlled YAML.
- Avoid embedding runner logic in configuration files.
