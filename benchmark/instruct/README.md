# `benchmark/instruct/`

Instruct-only leaderboard suites live here.

These suites target chat/instruction-tuned checkpoints, but use the plain upstream lm-eval prompt format by default so results stay comparable to the configured harness tasks.

## Files

- [`mmlu_pro.yaml`](./mmlu_pro.yaml): MMLU-Pro direct-answer 5-shot multiple-choice protocol through the upstream `leaderboard_mmlu_pro` lm-eval task.
- [`gsm8k.yaml`](./gsm8k.yaml): GSM8K 8-shot Chain-of-Thought exact-match generation through the upstream `gsm8k_cot` lm-eval task.

## LM-Eval Contracts

- Instruct suites do not apply a chat template by default. If we switch to a chat-template protocol later, that should be done consistently across the instruct leaderboard and documented as a separate protocol.
- `mmlu_pro.yaml` uses `leaderboard_mmlu_pro`, which is a 5-shot multiple-choice/loglikelihood task with metric `acc`; it does not generate Chain-of-Thought and therefore does not set `max_gen_toks`.
- `gsm8k.yaml` uses the upstream `gsm8k_cot` task, whose Chain-of-Thought prompt uses 8 few-shot exemplars; this is the common GSM8K CoT protocol rather than the non-CoT upstream `gsm8k` task's 5-shot setting.
