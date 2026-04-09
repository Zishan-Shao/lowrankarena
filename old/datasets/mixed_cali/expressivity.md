Nice, this is the right time to nail down a *story* for your training data, not just “we used Alpaca because SVD-LLM did”.

I’d think in terms of **two regimes** you can explicitly propose in the paper:

1. **Fair apples-to-apples with SVD-LLM / SAES-SVD**
2. **Expressivity-oriented multi-task mixture** (your actual contribution)

I’ll suggest concrete datasets for each “bucket”.

---

## 1. “Fair” setup: match SVD-LLM as closely as possible

For any claims like “we improve over SVD-LLM under the same data”, you want something very close to what they do:

* **Calibration / whitening dataset**
  SVD-LLM and several follow-ups pick **256 sentences from WikiText-2** as calibration data for whitening + SVD.

  * Use: WikiText-2 **train** split, randomly sample 256 sequences (your current setup already does something like this).

* **Parameter-update / SLRA / LoRA data**
  SVD-LLM uses **Alpaca 50k** for the low-rank recovery step.

  * Use: `yahma/alpaca-cleaned` (or similar cleaned Alpaca variant).

So one training regime you should propose is literally:

> **Regime A (SVD-LLM-style):**
> Calibration: 256 sentences from WikiText-2 train.
> LoRA / SLRA update data: 50k Alpaca-style instruction pairs (Alpaca-cleaned).
> Evaluation: WikiText-2, C4, ARC_e/c, OpenBookQA, HellaSwag, WinoGrande, PIQA, MathQA, etc. (same as SVD-LLM / SAES-SVD).

That gives you a clean “we did everything they did, just swapped the compression method” row.

---

## 2. Expressivity setup: mixed datasets by *skill bucket*

For your **actual method** (activation-LoRA, mixed objectives), I’d explicitly propose a **multi-task training set** that’s *different from the eval benchmarks* but **covers the same skills**.

Think by buckets:

### Bucket A – General LM (PPL on WikiText2, PTB, C4)

Goal: keep the compressed model a decent language model.

**Train on:**

* **WikiText-2 train** (Merity et al.) – same family as your eval `wikitext2_val`.
* **PTB / Penn Treebank** text variant (e.g., `ptb_text_only`) – matches PTB eval distribution.
* Optional: small **C4** subset, since C4 is also a standard LM benchmark used in SVD-LLM / SVD-LLM-V2 / GRASP.

**How to use:**

* Full-sequence LM loss: `labels = input_ids`.
* These batches are where you repair **LM PPL** across corpora.

### Bucket B – General instruction / QA / commonsense

Goal: support OpenBookQA, ARC_e/c, HellaSwag, PIQA, WinoGrande-style reasoning **without training directly on those test sets**.

**Datasets you can cite and sample from:**

* **Alpaca-cleaned** (as above).
* One or two additional open instruction collections, e.g.:

  * `Muennighoff/natural-instructions` (multi-task instructions)
  * A curated instruction dataset from an “awesome instruction datasets” list.

Optionally, include generic QA/commonsense sources (NOT the exact eval benchmarks):

* Something in the spirit of TriviaQA / CommonsenseQA / Natural Questions (or any open QA datasets you like).

**How to use:**

* Standard SFT formatting (like you already do with your `Prompter("alpaca")` logic).
* Mask user part in labels (Alpaca-style).

This bucket keeps **instruction following + commonsense** alive after heavy compression, so your model doesn’t become just a raw LM.

### Bucket C – Math & symbolic reasoning (for MathQA / GSM8K-style eval)

Goal: get the math circuits going **without directly training on MathQA test data**.

**Good sources:**

* **GSM8K** – standard grade-school math word problems; widely used as math-reasoning benchmark and training source.
* **AQuA-RAT** – multiple-choice math word problems with rationales; MathQA is built by re-annotating AQuA-RAT.
* Optional synthetic / curated math SFT sets (there are several open math-instruction datasets; you can cite a math-word-problem collection repo).

Since **MathQA itself** is your *evaluation* dataset, safest is:

* **Don’t** train on MathQA at all in your “main” setting,
* Or if you *do* use it, restrict to the **train split only** and clearly label that variant as “MathQA-tuned / oracle”.

**How to use:**

* Format as instruction SFT (“Solve this math problem: …”) with answer as output; optionally include chain-of-thought where available.

---

## 3. A concrete mixture you can propose

In the paper, you can write something like:

> **Regime B (multi-task mixture):**
> We construct a multi-task training set (D_{\text{mix}}) comprising:
>
> * **LM data (40%)**: WikiText-2 train, PTB train, and a 50k-sample subset of C4.
> * **Instruction/QA data (40%)**: 50k Alpaca-cleaned examples and 50k mixed general instructions from Natural-Instructions and related corpora.
> * **Math reasoning data (20%)**: 30k word-problem examples from GSM8K and AQuA-RAT-style datasets (excluding MathQA test data).
>
> From (D_{\text{mix}}), we sample training batches by first choosing the bucket according to these proportions and then drawing a random example from that bucket. For LM batches we apply a full-sequence next-token loss; for instruction and math batches we use standard instruction-tuning loss with masked labels on the prompt.

You can implement it either as:

* **Two-stage schedule**:

  1. Stage 1 – LM repair on Bucket A (full-seq loss);
  2. Stage 2 – SFT on Buckets B+C.

* Or **per-batch mixture sampling**: choose LM vs SFT vs Math by probability and feed into the same `train_act_lora_full_seq` / `train_act_lora` loop.

Either way, this gives a clear, defensible answer when a reviewer asks:

> “What data did you use to train your LoRA adapters, and why should I believe your method retains general capabilities across LM / QA / math?”

---

## 4. How I’d phrase the “dataset story” in your paper

You can summarize your design like this:

1. **Calibration (all methods):**

   * 256 WikiText-2 train sentences, seqlen 2048, following ASVD/SVD-LLM conventions.

2. **SVD-LLM-style baseline training:**

   * Alpaca-cleaned (50k) only, as in original SVD-LLM.

3. **Our multi-task training:**

   * Bucketed mix: LM (WikiText-2, PTB, C4), general instructions (Alpaca + other instruction datasets), math QA (GSM8K / AQuA-RAT, etc.), with stated proportions.

Then you can have tables where:

* One block compares **your method vs SVD-LLM under the Alpaca-only regime** (clean fairness).
* Another block shows **how much further your method can go under the multi-task mixture**, with clear note that this uses more diverse training data.

If you want, I can help you turn this into a small table like:

| Bucket | Datasets | Loss | Purpose |
| ------ | -------- | ---- | ------- |

that you can drop straight into the methodology section.
