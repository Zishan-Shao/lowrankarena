# Calibration Rank Stability

This audit recomputes the math-retention ranking for Llama-3.1-8B at keep ratio
0.8 across four method-independent calibration settings: WikiText primary,
WikiText draw s0, WikiText draw s1, and C4 primary. It uses the complete MathQA
and MMLU-Math result files already hosted on HF; it does not rebuild or
re-evaluate a checkpoint.

The reported `math_retention` is

```text
0.5 * (MathQA / dense_MathQA + MMLU-Math / dense_MMLU-Math)
```

It is intentionally a two-task math slice, not the paper's five-component
Quality Retention score.

## Finding

The ordering is identical in all four settings:

```text
MoDeGPT > SVD-LLM v1 > Basis Sharing > ASVD
```

The maximum rank displacement is `0` on this backbone, budget, calibration,
and metric slice. Observed math-retention ranges are:

| Method | Minimum | Maximum | Range | Ranks |
| --- | ---: | ---: | ---: | --- |
| MoDeGPT | 0.866362 | 0.900163 | 0.033801 | 1, 1, 1, 1 |
| SVD-LLM v1 | 0.654208 | 0.662065 | 0.007857 | 2, 2, 2, 2 |
| Basis Sharing | 0.615860 | 0.649806 | 0.033946 | 3, 3, 3, 3 |
| ASVD | 0.512313 | 0.585540 | 0.073227 | 4, 4, 4, 4 |

This is an observed bounded result, not a universal claim that calibration
data can never change a ranking.

## Reproduce

Install the lightweight HF CLI if it is not already available:

```bash
python -m pip install 'huggingface_hub[cli]'
hf auth login  # only needed when gated access is not already configured
```

From the repository root:

```bash
hf download Duke-CEI-SVD/LowRankArena \
  --repo-type model \
  --revision 221e1809a8054b08788d477f7a732fcbfaa7c456 \
  --include 'rebuttal/calibration_stability_w2_full_retention_math_20260727/results/*.json' \
  --local-dir hf_snapshot

python audits/calibration_data_sensitivity/rank_stability/analyze.py \
  hf_snapshot/rebuttal/calibration_stability_w2_full_retention_math_20260727/results
```

The analyzer uses only the Python standard library. It checks the 17 expected
result files, recomputes every score and rank, compares them with
[`expected.csv`](./expected.csv), and prints a compact summary. A successful
run ends with `PASS`.

## Provenance

- Result snapshot revision:
  [`221e1809a8054b08788d477f7a732fcbfaa7c456`](https://huggingface.co/Duke-CEI-SVD/LowRankArena/tree/221e1809a8054b08788d477f7a732fcbfaa7c456/rebuttal/calibration_stability_w2_full_retention_math_20260727/results)
- Checkpoints referenced by the result records:
  `877cfce2e2723597969c52139f38db1b16c08543`
- Evaluation backend: `lm-eval-harness 0.4.11`
- Evaluation: full MathQA and MMLU-Math splits, zero-shot, batch size 1, FP16
