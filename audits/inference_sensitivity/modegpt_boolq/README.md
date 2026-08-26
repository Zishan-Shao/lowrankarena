# MoDeGPT BoolQ Floor Audit

This audit reproduces the anomalous BoolQ values for MoDeGPT-compressed
Llama-3.1-8B from the complete normalized result files. It distinguishes a
50% random-choice reference from the empirical majority-label baseline
`2033 / 3270 = 0.6217125382`.

## Finding

| Model | Keep | MCQ macro | BoolQ accuracy | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Dense | 1.0 | 0.715470 | 0.830581 | above majority baseline |
| MoDeGPT | 0.4 | 0.381388 | 0.398777 | below random reference |
| MoDeGPT | 0.5 | 0.439357 | 0.582263 | between random and majority |
| MoDeGPT | 0.6 | 0.503769 | 0.621713 | exactly majority baseline |
| MoDeGPT | 0.7 | 0.566771 | 0.621713 | exactly majority baseline |
| MoDeGPT | 0.8 | 0.589620 | 0.411621 | below random reference |

The non-monotonic BoolQ values, including keep 0.8 below the 50% reference,
should be interpreted as a task-level floor and prediction/label-bias effect.
They are not evidence of a meaningful "negative capability." The seven-task
MCQ macro and perplexity remain more informative aggregate signals.

## Fast reproduction from published results

This path needs no GPU and downloads only six JSON files:

```bash
python -m pip install 'huggingface_hub[cli]'
hf auth login  # only needed when gated access is not already configured

hf download Duke-CEI-SVD/LowRankArena \
  --repo-type model \
  --revision db12fe2017e9075d5c0c46f80b6ce2c9ccb431dd \
  --include \
    'results/evaluation/llama31_8b/mcq/llama31-8b-dense.json' \
    'results/evaluation/llama31_8b/mcq/llama31-8b-modegpt-*.json' \
  --local-dir hf_snapshot

python audits/inference_sensitivity/modegpt_boolq/extract.py \
  hf_snapshot/results/evaluation/llama31_8b/mcq
```

The standard-library extractor checks the backend, full-split sample count,
all six values, random/majority references, and
[`expected.csv`](./expected.csv). A successful run ends with `PASS`.

## Optional full re-evaluation

This reruns BoolQ for keep 0.8 without regenerating the checkpoint. It requires
an accepted Llama license, roughly 13 GB of checkpoint storage, a CUDA GPU,
`transformers==4.57.6`, and `lm-eval==0.4.11`.

```bash
hf download Duke-CEI-SVD/LowRankArena \
  --repo-type model \
  --revision eb641cb9b28ac6d5706d9903dcc031d561593baf \
  --include 'checkpoints/low_rank/llama31_8b/modegpt/default_0.8/**' \
  --local-dir hf_checkpoint

CUDA_VISIBLE_DEVICES=0 lm-eval run \
  --model hf \
  --model_args pretrained=hf_checkpoint/checkpoints/low_rank/llama31_8b/modegpt/default_0.8,dtype=float16,trust_remote_code=True \
  --tasks boolq \
  --num_fewshot 0 \
  --batch_size 1 \
  --device cuda:0 \
  --output_path boolq_keep80
```

The expected full-split result is `0.4116207951` accuracy over 3,270 examples
with standard error `0.0086073577`. Minor package or numerical differences
should be reported rather than silently replacing the pinned result.

## Provenance

- Canonical result revision:
  [`db12fe2017e9075d5c0c46f80b6ce2c9ccb431dd`](https://huggingface.co/Duke-CEI-SVD/LowRankArena/tree/db12fe2017e9075d5c0c46f80b6ce2c9ccb431dd/results/evaluation/llama31_8b/mcq)
- Reorganized keep-0.8 checkpoint:
  [`eb641cb.../checkpoints/low_rank/llama31_8b/modegpt/default_0.8`](https://huggingface.co/Duke-CEI-SVD/LowRankArena/tree/eb641cb9b28ac6d5706d9903dcc031d561593baf/checkpoints/low_rank/llama31_8b/modegpt/default_0.8)
- Evaluation: `lm-eval-harness 0.4.11`, zero-shot, full BoolQ validation
  split, batch size 1, FP16 inference
