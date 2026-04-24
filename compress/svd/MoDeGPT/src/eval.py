import logging
import math
import time

import numpy as np
import torch
from datasets import load_dataset


from src.adapters.model_adapter import MLPComponents, ModelAdapter

import random

logger = logging.getLogger("MoDeGPT")


def load_c4(tokenizer, texts, n_samples):
    random.seed(1234)
    seqlen = 2048
    calib_samples = []
    while len(calib_samples) < n_samples:
        while True:
            i = random.randint(0, len(texts) - 1)
            tok = tokenizer(texts[i], return_tensors="pt")
            if tok.input_ids.shape[1] > seqlen:
                break
        start = random.randint(0, tok.input_ids.shape[1] - seqlen - 1)
        calib_samples.append(tok.input_ids[:, start : start + seqlen].to(device="cuda"))

    return calib_samples


def load_calibration_texts(calib_size, model, tokenizer, batch_size: int, dataset="wikitext"):
    if dataset == "alpaca":
        return load_alpaca_texts(
            calib_size=calib_size, model=model, tokenizer=tokenizer, batch_size=batch_size
        )

    if dataset == "wikitext":
        dataset_obj = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        joined_texts = "\n\n".join(dataset_obj["text"])
    elif dataset == "c4":
        dataset_obj = load_dataset(
            "json",
            data_files={
                "train": "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"
            },
        )
        texts = [t for t in dataset_obj["train"]["text"] if len(t.strip()) > 0]
        joined_texts = "\n\n".join(texts[:10000])  # Limit to avoid memory issues
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'wikitext' or 'c4'")

    chunked = chunk_text(
        model=model, tokenizer=tokenizer, long_texts=joined_texts, min_threshold=2048
    )

    np.random.seed(1234)
    num_chunks = chunked.shape[0]
    indices = np.random.choice(num_chunks, size=min(int(calib_size), num_chunks), replace=False)

    batches = []
    for i in range(0, len(indices), batch_size):
        batch_ind = indices[i : i + batch_size]
        batch_tensor = chunked[batch_ind]
        batches.append(batch_tensor.to("cuda"))

    return batches


def load_alpaca_texts(calib_size, model, tokenizer, batch_size, seed=1234):
    def format_sample(sample):
        if sample.get("input"):
            return (
                f"Below is an instruction that describes a task, paired with an input that provides further context. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{sample['instruction']}\n\n"
                f"### Input:\n{sample['input'] if 'input' in sample else ''}\n\n"
                f"### Response:\n"
            ) + tokenizer.eos_token
        else:
            return (
                f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{sample['instruction']}\n\n"
                f"### Response:\n"
            ) + tokenizer.eos_token

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    max_length = min(2048, model.config.max_position_embeddings)

    dataset = dataset.shuffle(seed=seed)

    n_tokens = max_length * calib_size
    tokens = []

    for sample in dataset:
        text = format_sample(sample)

        # ids of shape [num_tokens of text]
        ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
        tokens.extend(ids)

        if len(tokens) >= n_tokens:
            break

    tokens = tokens[:n_tokens]

    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.view(calib_size, max_length)

    batches = []
    for i in range(0, calib_size, batch_size):
        batch_tensor = tokens[i : i + batch_size]
        batches.append(batch_tensor.to(device="cuda"))

    # batched_tokens = tokens.view(-1, batch_size, max_length).to(device="cuda")

    return batches


def chunk_text(model, tokenizer, long_texts: str, min_threshold):
    input_ids = tokenizer(
        long_texts, truncation=False, return_tensors="pt", add_special_tokens=False
    )["input_ids"]

    max_length = min(2048, model.config.max_position_embeddings)

    n_chunks = input_ids[0].size(0) // max_length
    chunked = input_ids[0][: n_chunks * max_length].view(-1, max_length)
    return chunked


@torch.no_grad()
def compute_perplexity(
    model, tokenizer, bs=16, device="cuda", dataset="wikitext", adapter: ModelAdapter = None
):

    model.eval()

    if dataset == "wikitext":
        dataset_obj = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset_obj["text"])
    elif dataset == "c4":
        dataset_obj = load_dataset(
            "json",
            data_files={
                "validation": "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation.00000-of-00008.json.gz"
            },
        )
        texts = [t for t in dataset_obj["validation"]["text"] if len(t.strip()) > 0]
        text = "\n\n".join(texts[:5000])
    elif dataset == "alpaca":
        texts = get_alpaca_eval_data()
        text = "\n\n".join(texts)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'wikitext', 'c4', or 'alpaca'")
    testenc = tokenizer(text, return_tensors="pt")
    testenc = testenc.input_ids

    seqlen = 2048

    nsamples = min(testenc.numel() // seqlen, 512)

    nlls = []
    total_tokens_processed = 0
    print(f"nsamples {nsamples}")

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    for i in range(0, nsamples, bs):
        elapsed_so_far = time.perf_counter() - t_start
        if nlls:
            running_ppl = torch.exp(torch.stack(nlls).sum() / (total_tokens_processed)).item()
            tps = total_tokens_processed / elapsed_so_far if elapsed_so_far > 0 else 0
            print(
                f"\rsample {i}/{nsamples} | ppl: {running_ppl:.2f} | {tps:,.0f} tok/s | {elapsed_so_far:.1f}s    ",
                end="",
                flush=True,
            )
        else:
            print(f"\rsample {i}/{nsamples}", end="", flush=True)

        j = min(i + bs, nsamples)

        inputs = testenc[:, (i * seqlen) : (j * seqlen)].to(device)
        inputs = inputs.reshape(j - i, seqlen)

        lm_logits = model(inputs).logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        neg_log_likelihood = loss.float() * (seqlen - 1) * (j - i)

        nlls.append(neg_log_likelihood)

        total_tokens_processed += (j - i) * seqlen

    torch.cuda.synchronize()
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    tokens_per_sec = total_tokens_processed / elapsed
    print(f"\n── Throughput ──")
    print(f"  Total tokens : {total_tokens_processed:,}")
    print(f"  Wall time    : {elapsed:.2f} s")
    print(f"  Throughput   : {tokens_per_sec:,.0f} tok/s ({tokens_per_sec / 1000:.1f} ktok/s)")

    if adapter:
        adapter.metrics["throughput_tok/s"] = tokens_per_sec
        adapter.metrics["throughput_ktok/s"] = tokens_per_sec / 1000

    # Compute perplexity
    # ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (seqlen - 1)))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


def get_alpaca_eval_data(n_samples: int = 500):
    # tatsu-lab/alpaca only has a "train" split, so we hold out the
    # last `n_samples` examples as a pseudo-test set.
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.select(range(len(dataset) - n_samples, len(dataset)))

    formatted_texts = []
    for example in dataset:
        if example.get("input", ""):
            text = (
                f"Below is an instruction that describes a task, paired with an input that provides further context. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                f"### Response:\n{example['output']}"
            )
        else:
            text = (
                f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Response:\n{example['output']}"
            )

        formatted_texts.append(text)

    return formatted_texts


@torch.no_grad()
def evaluate_perplexity_alpaca(model, tokenizer, device="cuda"):
    texts = get_alpaca_eval_data()

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    max_length = 2048

    print(f"Evaluating Perplexity on {len(texts)} samples...")

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(
            device
        )

        labels = inputs["input_ids"].clone()

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        seq_len = inputs["input_ids"].size(1)

        if not torch.isfinite(loss):
            print("Warning: Non-finite loss detected")
            continue

        total_loss += loss.item() * seq_len
        total_tokens += seq_len

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_loss / total_tokens
    ppl = math.exp(avg_nll)

    print(f"Alpaca Perplexity: {ppl:.2f}")
    return ppl
