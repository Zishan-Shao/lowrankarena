# `benchmark/instruct/`

Instruct-only leaderboard and appendix suites live here.

These suites target chat/instruction-tuned checkpoints, but use the plain upstream lm-eval prompt format by default so results stay comparable to the configured harness tasks.

## Files

- [`mmlu_pro.yaml`](./mmlu_pro.yaml): MMLU-Pro direct-answer 5-shot multiple-choice protocol through the upstream `leaderboard_mmlu_pro` lm-eval task.
- [`gsm8k.yaml`](./gsm8k.yaml): GSM8K 8-shot Chain-of-Thought exact-match generation through the upstream `gsm8k_cot` lm-eval task.
- [`aime.yaml`](./aime.yaml): AIME 2024 hard-math stress test through the upstream `aime24` task, with solved count derived from exact-match accuracy and sample count.
- [`ifeval.yaml`](./ifeval.yaml): IFEval instruction-following sanity suite with strict and loose prompt-/instruction-level metrics.

## LM-Eval Contracts

- Instruct suites do not apply a chat template by default. If we switch to a chat-template protocol later, that should be done consistently across the instruct leaderboard and documented as a separate protocol.
- `mmlu_pro.yaml` uses `leaderboard_mmlu_pro`, which is a 5-shot multiple-choice/loglikelihood task with metric `acc`; it does not generate Chain-of-Thought and therefore does not set `max_gen_toks`.
- `gsm8k.yaml` uses the upstream `gsm8k_cot` task, whose Chain-of-Thought prompt uses 8 few-shot exemplars; this is the common GSM8K CoT protocol rather than the non-CoT upstream `gsm8k` task's 5-shot setting.
- `aime.yaml` uses upstream `aime24`, greedy generation, `num_fewshot: 0`, and upstream exact-match extraction. The normalized result records `metrics.solved_count`, `metrics.sample_count`, and `metrics.solved_accuracy` in addition to the primary `exact_match`.
- `ifeval.yaml` is the exception to the no-chat-template default: because it evaluates instruction following, it explicitly sets `apply_chat_template: true` for a unified instruct protocol. Upstream `ifeval` fixes `max_gen_toks: 1280`, `temperature: 0.0`, and `do_sample: false`.
