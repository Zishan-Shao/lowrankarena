# Calibration-Set Source Audit

This audit verifies the four method-independent, source-indexed calibration
inputs used by the rank-stability matrix. Corpus units are selected before any
compression method runs; no compared method's loader or random seed
participates in selecting a document or token window.

| Setting | Source | Samples | Tokens/sample | Tensor-byte SHA-256 |
| --- | --- | ---: | ---: | --- |
| WikiText primary | WikiText-2 raw train articles | 256 | 2,048 | `72b25ac4b3e08f9fbbabc71e00f3bb5ec32a60a3ae96be67289808f25d6669ce` |
| WikiText s0 | WikiText-2 raw train articles | 256 | 2,048 | `395153e0199b10ddd1dad782f59fc8a9a0f694eea53429f07fcead46902268ba` |
| WikiText s1 | WikiText-2 raw train articles | 256 | 2,048 | `79f90ff9c25a75b1fe6565d6b6ff2e178e168ef65564076066f08316cb0c6b5f` |
| C4 primary | pinned C4 English train-shard documents | 256 | 2,048 | `59443b6b553b4ba9920415b7a5fdfcf248201789fe53e01b71432db446aa87d5` |

Each sample comes from a distinct source article/document. Selection uses
public SHA-256 salts, no PRNG, the Llama-3.1-8B tokenizer at revision
`d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`, and no special tokens. Numeric
draw labels are input-selection labels, not method RNG seeds.

## Reproduce

Install the HF CLI and authenticate if required:

```bash
python -m pip install 'huggingface_hub[cli]'
hf auth login
```

Download only the protocol plus the four inputs used by the matrix:

```bash
hf download Duke-CEI-SVD/LowRankArena \
  --repo-type model \
  --revision 221e1809a8054b08788d477f7a732fcbfaa7c456 \
  --include \
    'rebuttal/new_checkpoints/calibration_corpora/v1/README.md' \
    'rebuttal/new_checkpoints/calibration_corpora/v1/SELECTION_PROTOCOL.md' \
    'rebuttal/new_checkpoints/calibration_corpora/v1/wikitext/**' \
    'rebuttal/new_checkpoints/calibration_corpora/v1/wikitext_draws/s0/**' \
    'rebuttal/new_checkpoints/calibration_corpora/v1/wikitext_draws/s1/**' \
    'rebuttal/new_checkpoints/calibration_corpora/v1/c4/**' \
  --local-dir hf_snapshot

python audits/calibration_data_sensitivity/calibration_source/verify.py \
  hf_snapshot/rebuttal/new_checkpoints/calibration_corpora/v1
```

The verifier uses only the Python standard library. For every input it checks:

- all files listed by `SHA256SUMS`;
- manifest schema, train-only source, sample count, sequence length, tokenizer,
  lack of method-loader/seed selection, and published tensor hash;
- NumPy header, `int64[256, 2048]` C-order layout, full tensor-byte hash, and
  every ordered row hash;
- 256 valid JSON records in both `selected_corpus.jsonl` and
  `source_indices.jsonl`.

A successful run ends with `PASS`. It verifies the exact released inputs and
does not regenerate them.

## Provenance

- Frozen bundle:
  [`calibration_corpora/v1` at `221e180...`](https://huggingface.co/Duke-CEI-SVD/LowRankArena/tree/221e1809a8054b08788d477f7a732fcbfaa7c456/rebuttal/new_checkpoints/calibration_corpora/v1)
- WikiText dataset: `wikitext/wikitext-2-raw-v1`, train split, fingerprint
  `7c4dea6941cc4a0a`
- C4 dataset: `allenai/c4`, revision
  `607bd4c8450a42878aa9ddc051a65a055450ef87`, train shard
  `en/c4-train.00000-of-01024.json.gz`

Earlier global-concatenation calibration bundles are deprecated and are not
accepted by this verifier.
