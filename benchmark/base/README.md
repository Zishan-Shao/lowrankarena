# `benchmark/base/`

Base-only evaluation suites live here.

These suites should avoid chat templates, instruction-following assumptions, and long-form generation contracts unless the suite explicitly documents a short-answer extraction protocol. They are meant to measure capability retention for pretrained/base checkpoints after compression.

## Files

- [`base_math.yaml`](./base_math.yaml): base-model math suite using 0-shot `lra_mathqa` plus local 5-shot `MMLU_Math`, both through MCQ/loglikelihood-style `lm-eval-harness` protocols suitable for non-instruct checkpoints.

## Contract

- Do not apply a chat template.
- Prefer loglikelihood or short-answer protocols over long Chain-of-Thought generation.
- Keep base-only suites separate from instruct generation suites so paper-facing base results are not coupled to instruction-format behavior.

## MathQA

The pinned `lm_eval==0.4.11` installation exposes an upstream `mathqa` task, but in the current `datasets==4.4.2` environment it fails at runtime because the Hugging Face `math_qa` dataset still depends on a dataset loading script. `lm-eval validate --tasks mathqa` can pass while an actual run fails with `Dataset scripts are no longer supported`.

LowRankArena therefore defines `lra_mathqa` under [`tasks/`](./tasks/lra_mathqa.yaml). It downloads the official `MathQA.zip` archive, verifies its SHA-256 checksum, and exposes train/dev/test splits through a local custom dataset function. `base_math.yaml` includes this task through `eval.include_paths`.

## MMLU_Math

`base_math.yaml` uses the local `mmlu_math` group under [`tasks/`](./tasks/mmlu_math.yaml). This group expands to the MMLU math subjects with 5-shot multiple-choice/loglikelihood scoring and deliberately avoids generative prompts, so it remains suitable for base checkpoints.
