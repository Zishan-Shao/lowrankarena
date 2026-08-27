# SAES-SVD reproduction attempt

This directory contains an **unofficial, independent reproduction attempt** of
[SAES-SVD](https://arxiv.org/abs/2602.03051), implemented from the public paper
description. It is not source code from the SAES-SVD authors and should not be
presented as their official implementation.

Under the tested LLaMA-7B configuration, this implementation did not recover
the performance reported for the closest corresponding setting in the paper.
For that reason, LowRankArena does not include these measurements in its main
matched leaderboard. We publish the implementation and negative reproduction
result so that the assumptions, commands, and possible discrepancies can be
inspected and corrected.

## Scope

The release intentionally contains only the paper-aligned reproduction path:

- [`saes_svd.py`](./saes_svd.py): layer-wise CEALC/ACES compression and
  checkpoint serialization.
- [`eval_lm_eval_saes.py`](./eval_lm_eval_saes.py): checkpoint reconstruction,
  perplexity evaluation, and `lm-eval-harness` evaluation.
- [`evaluater.py`](./evaluater.py) and [`utils/data_utils.py`](./utils/data_utils.py):
  the evaluation helpers used by the recorded run.
- [`OBSERVED_RESULTS.md`](./OBSERVED_RESULTS.md): the tested configuration,
  observed measurements, and the closest paper-reported reference.
- [`SOURCE_PROVENANCE.md`](./SOURCE_PROVENANCE.md): source origin and snapshot
  limitations.

No checkpoint, model weight, calibration cache, dataset copy, generated tensor,
or run cache is included. Experimental SAES-OPT output-refit and activation-LoRA
follow-ups are also excluded because they are extensions rather than the
paper-aligned baseline implemented here.

## Environment

Install LowRankArena and its evaluation dependencies, or use the compact
environment record in [`requirements.txt`](./requirements.txt):

```bash
python -m pip install -r compress/svd/SAES-SVD-reproduction/requirements.txt
```

The historical environment used for the March 2026 measurement was not frozen
at run time. The versions in `requirements.txt` record the later `flashsvd`
environment from which this release snapshot was prepared; this limitation is
part of the reproduction record.

Access to the base model may require Hugging Face authentication and acceptance
of the model's license and access conditions.

## Reproduction configuration

The recorded run used `jeffwan/llama-7b-hf`, a uniform parameter keep ratio of
0.4 for factorized linear layers, WikiText-2 training text for calibration,
1,024 calibration sequences of length 2,048, seed 42, and randomized SVD with
eight power iterations.

From the repository root:

```bash
python compress/svd/SAES-SVD-reproduction/saes_svd.py \
  --model_id jeffwan/llama-7b-hf \
  --output_dir /path/to/saes_llama7b_keep40 \
  --compression_ratio 0.4 \
  --seq_len 2048 \
  --calib_sequences 1024 \
  --batch_size 1 \
  --seed 42 \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --dataset_split train \
  --max_tokens_total 524288 \
  --beta_mode aces \
  --aces_objective ratio \
  --svd_method randomized \
  --svd_niter 8 \
  --device cuda \
  --teacher_device cuda \
  --stats_device cpu \
  --compute_device cuda \
  --dtype float16 \
  --teacher_dtype float16 \
  --factor_dtype float32 \
  --compute_dtype float32
```

Evaluate the serialized checkpoint with the same task family recorded in the
reproduction notes:

```bash
python compress/svd/SAES-SVD-reproduction/eval_lm_eval_saes.py \
  --base_model jeffwan/llama-7b-hf \
  --ckpt_dir /path/to/saes_llama7b_keep40 \
  --device cuda \
  --dtype float16 \
  --factor_dtype float32 \
  --run_ppl \
  --ppl_datasets wikitext2,ptb,c4_stream \
  --ppl_seq_len 2048 \
  --ppl_batch_size 4 \
  --c4_val_stream on \
  --c4_val_docs 2000 \
  --c4_val_dataset allenai/c4 \
  --run_lm_eval \
  --tasks openbookqa,arc_easy,arc_challenge,winogrande,hellaswag,piqa,mathqa \
  --num_fewshot 0 \
  --batch_size 4 \
  --out_json /path/to/saes_llama7b_keep40/lm_eval_7tasks.json
```

The commands above reconstruct the recorded argument set from the run notes and
script defaults. Because the original shell transcript and frozen environment
were not retained, they should be read as the best available reconstruction,
not as a byte-for-byte archived launch record.

## Interpretation

The result in this directory supports only the following narrow statement:
this independent implementation did not reproduce the paper-reported
performance for the tested LLaMA-7B budget. It does **not** establish that the
SAES-SVD method is incorrect or generally irreproducible. Differences may arise
from implementation details, budget semantics, calibration sampling, model
conversion, dependency versions, or evaluation protocol.

Corrections and clarifications from the paper authors or the community are
welcome.

## License

The LowRankArena-authored reproduction code in this directory is covered by the
repository's [MIT License](../../../LICENSE). The SAES-SVD paper and all base
models and datasets retain their own licenses and usage conditions.
