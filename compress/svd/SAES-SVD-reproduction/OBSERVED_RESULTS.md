# Observed SAES-SVD reproduction results

## Tested configuration

- Run date: 2026-03-05
- Base model: `jeffwan/llama-7b-hf`
- Uniform parameter keep ratio: `0.4`
- Calibration sequences: `1024`
- Sequence length: `2048`
- Maximum calibration tokens: `524288`
- Random seed: `42`
- Randomized-SVD power iterations: `8`
- Evaluation tasks: OpenBookQA, ARC-Easy, ARC-Challenge, WinoGrande,
  HellaSwag, PIQA, and MathQA, all zero-shot

In the released script, `--compression_ratio 0.4` is a parameter **keep**
ratio: for an `m x n` weight, the rank is
`floor(m * n * 0.4 / (m + n))`. The SAES-SVD paper instead labels increasingly
aggressive settings with increasing compression ratios. The closest paper row
is therefore its `0.6` compression-ratio setting, not its `0.4` row.

## Primary reproduction

| Measurement | This reproduction | Paper reference |
| --- | ---: | ---: |
| WikiText-2 perplexity | 31.5990 | 22.01 |
| PTB perplexity | 244.6919 | 116.83 |
| C4 perplexity | 130.9742 | 93.97 |
| Seven-task average accuracy | 0.3036 | 0.34 |

The paper-reference values are from Table 1 of
[arXiv:2602.03051v1](https://arxiv.org/abs/2602.03051), LLaMA-7B at its reported
`0.6` compression ratio. The protocols are not guaranteed to be identical, so
the table documents a reproduction gap rather than an exact controlled
replication failure.

The six-task normalized-accuracy average used in the original local run note
was `0.3018`; WinoGrande was excluded because that result did not expose an
`acc_norm` field.

## Memory observed during evaluation

- Model-weight allocation: `5590.36 MiB`
- Peak allocated GPU memory: `7850.63 MiB`
- Peak reserved GPU memory: `9190.00 MiB`

## Diagnostic follow-ups not released as the baseline

Two non-paper extensions were also tested locally and are recorded here only
for completeness. Their source is intentionally excluded from this baseline
release.

| Variant | WikiText-2 PPL | PTB PPL | C4 PPL | Avg. accuracy (7) |
| --- | ---: | ---: | ---: | ---: |
| SAES-OPT strict (teacher, no output refit) | 31.1091 | 252.4593 | 130.9742 | 0.3055 |
| SAES-OPT plus (teacher + output refit) | 59.9643 | 777.6294 | 295.1548 | 0.3066 |

These extensions are not treated as SAES-SVD paper results and are not part of
the LowRankArena leaderboard.
