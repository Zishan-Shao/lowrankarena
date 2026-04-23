# `benchmark/base/`

Base-only evaluation suites live here.

These suites should avoid chat templates, instruction-following assumptions, and long-form generation contracts unless the suite explicitly documents a short-answer extraction protocol. They are meant to measure capability retention for pretrained/base checkpoints after compression.

## Files

- [`base_math.yaml`](./base_math.yaml): base-model math suite using `lra_mathqa` plus upstream `mmlu_stem`, both through MCQ/loglikelihood-style `lm-eval-harness` protocols suitable for non-instruct checkpoints.

## Contract

- Do not apply a chat template.
- Prefer loglikelihood or short-answer protocols over long Chain-of-Thought generation.
- Keep base-only suites separate from instruct generation suites so paper-facing base results are not coupled to instruction-format behavior.

## MathQA

The pinned `lm_eval==0.4.11` installation exposes an upstream `mathqa` task, but in the current `datasets==4.4.2` environment it fails at runtime because the Hugging Face `math_qa` dataset still depends on a dataset loading script. `lm-eval validate --tasks mathqa` can pass while an actual run fails with `Dataset scripts are no longer supported`.

LowRankArena therefore defines `lra_mathqa` under [`tasks/`](./tasks/lra_mathqa.yaml). It downloads the official `MathQA.zip` archive, verifies its SHA-256 checksum, and exposes train/dev/test splits through a local custom dataset function. `base_math.yaml` includes this task through `eval.include_paths`.

## MMLU-STEM

`base_math.yaml` uses the upstream `mmlu_stem` group from `lm-eval-harness==0.4.11`. This group expands to the official STEM subset of MMLU multiple-choice tasks and deliberately uses the default MCQ/loglikelihood protocol instead of a generative prompt, so it remains suitable for base checkpoints.
