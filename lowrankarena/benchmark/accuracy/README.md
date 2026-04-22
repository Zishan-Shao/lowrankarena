# `benchmark/accuracy/`

This directory contains accuracy-oriented benchmark suites.

Suites in this directory select their own evaluation backend through the `eval.backend` field. Most classification-style suites still route through `lm-eval-harness 0.4.11`, while perplexity-style suites may use a repo-owned runner when the preprocessing contract needs to stay fixed. The Python entrypoint is still [`scripts/run_eval.py`](../../scripts/run_eval.py), which dispatches to the configured backend and writes normalized outputs to [`results/eval/`](../../results/eval/README.md).

## Files

- [`mcq.yaml`](./mcq.yaml): multi-task multiple-choice suite for headline commonsense QA reporting.
- [`mmlu_pro.yaml`](./mmlu_pro.yaml): official `MMLU-Pro` Chain-of-Thought protocol through the upstream `mmlu_pro` lm-eval group. The normalized result uses the official weighted group aggregate for the headline score and keeps all 14 subject rows in `details.tasks`.
- [`gsm8k.yaml`](./gsm8k.yaml): `GSM8K` exact-match suite using the upstream `gsm8k` lm-eval task.
- [`ppl.yaml`](./ppl.yaml): unified perplexity suite over `wikitext2` test and a fixed-budget `c4_stream` validation stream, evaluated with the repo-owned contiguous-block runner.
- [`mmlu.yaml`](./mmlu.yaml): dedicated MMLU suite using the official `mmlu` group.

## Conventions

- Use canonical `lm-eval` task IDs for suites that target `lm_eval_harness`.
- Record metric preference in the suite config so downstream scoring stays deterministic.
- Keep environment caveats in the suite that requires them, not in the runner.
- Put task-local lm-eval protocol flags in the suite when they are part of the benchmark contract. Use the CLI only for ad hoc overrides such as `--limit` or a temporary device change.
- `mcq.yaml` keeps both `acc` and `acc_norm` in the normalized payload. The headline `mean` uses the preferred metric order from the suite config, so it reports `acc_norm` when available and falls back to `acc` for tasks that do not expose `acc_norm`.

## LM-Eval Contracts

- `mmlu_pro.yaml` does not apply a chat template. It uses the upstream `mmlu_pro` group, so the prompt, 5-shot CoT context, extraction regex, temperature, and stop sequence follow the installed harness task definition.
- `mmlu_pro.yaml` overrides only `max_gen_toks=256` at the suite layer so 2048-context LLaMA-7B checkpoints do not trip the HF backend assertion on the upstream task's `max_gen_toks: 2048`.
- `mmlu_pro.yaml` reads the official `groups.mmlu_pro` aggregate for the main score and keeps the 14-domain subject breakdown in the normalized task details for appendix reporting.
- `gsm8k.yaml` does not apply a chat template. It evaluates the upstream `gsm8k` lm-eval task under the pinned `lm_eval==0.4.11` environment.

## PPL Contract

- `ppl.yaml` does not use chat templates.
- Raw text is tokenized with the checkpoint tokenizer and `add_special_tokens=False`.
- Evaluation uses contiguous, non-overlapping blocks with one fixed `max_length`.
- The suite reports only evaluation-split perplexity.
- `wikitext2` uses the `test` split, and `c4_stream` uses the `validation` stream with a fixed token budget so runs stay comparable.
