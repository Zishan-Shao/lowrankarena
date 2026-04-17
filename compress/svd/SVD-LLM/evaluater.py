from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional dependency
    load_dataset = None


SCRIPT_DIR = Path(__file__).resolve().parent


def _require_datasets():
    if load_dataset is None:
        raise RuntimeError("datasets is required for SVD-LLM evaluation.")


def _encode_text(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return [int(token_id) for token_id in encoded["input_ids"]]


def _build_blocks(token_ids: list[int], seq_len: int) -> torch.Tensor:
    usable = (len(token_ids) // int(seq_len)) * int(seq_len)
    if usable < int(seq_len):
        raise ValueError(f"Need at least one full block of length {seq_len}, got {len(token_ids)} tokens.")
    return torch.tensor(token_ids[:usable], dtype=torch.long).view(-1, int(seq_len))


def _stream_c4_text(max_docs: int = 256) -> str:
    _require_datasets()
    docs = []
    stream = None
    last_error = None
    for path, name in (("allenai/c4", "en"), ("c4", "en")):
        try:
            stream = load_dataset(path, name, split="validation", streaming=True)
            break
        except Exception as exc:  # pragma: no cover - depends on local dataset access
            last_error = exc
    if stream is None:
        raise RuntimeError("Unable to load C4 validation split.") from last_error
    for item in itertools.islice(stream, max_docs):
        text = str(item.get("text", "") or "")
        if text.strip():
            docs.append(text)
    if not docs:
        raise ValueError("C4 stream returned no usable documents.")
    return "\n\n".join(docs)


def _load_token_ids(dataset_name: str, tokenizer) -> list[int]:
    _require_datasets()
    dataset_name = str(dataset_name).lower()
    if "wikitext2" in dataset_name or "wikitext" in dataset_name:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(str(item) for item in dataset["text"])
        return _encode_text(tokenizer, text)
    if "ptb" in dataset_name:
        dataset = load_dataset("ptb_text_only", "penn_treebank", split="test")
        text = "\n\n".join(str(item) for item in dataset["sentence"])
        return _encode_text(tokenizer, text)
    if "c4" in dataset_name:
        return _encode_text(tokenizer, _stream_c4_text())
    raise ValueError(f"Unsupported evaluation dataset: {dataset_name}")


@torch.inference_mode()
def ppl_eval(
    model,
    tokenizer,
    datasets=("wikitext2",),
    model_seq_len: int = 2048,
    batch_size: int = 4,
    device: str | torch.device = "cuda",
    label: str = "PPL",
):
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    results = {}
    for dataset_name in datasets:
        token_ids = _load_token_ids(dataset_name, tokenizer)
        blocks = _build_blocks(token_ids, model_seq_len)
        total_nll = 0.0
        total_tokens = 0
        for start in range(0, int(blocks.shape[0]), int(batch_size)):
            batch = blocks[start : start + int(batch_size)].to(device)
            outputs = model(input_ids=batch)
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = batch[:, 1:].contiguous()
            loss_sum = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                reduction="sum",
            )
            total_nll += float(loss_sum.item())
            total_tokens += int(labels.numel())
        results[str(dataset_name)] = math.exp(total_nll / float(total_tokens))

    print(f"{label}: {results}")
    return results


@torch.inference_mode()
def eff_eval(
    model,
    tokenizer,
    dataset: str = "wikitext2",
    original_len: int = 4,
    generated_len: int = 128,
    batch_size: int = 1,
    device: str | torch.device = "cuda",
):
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    token_ids = _load_token_ids(dataset, tokenizer)
    prompt_blocks = _build_blocks(token_ids, original_len)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    total_time = 0.0
    total_generated_tokens = 0
    weight_memory = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    peak_memory = 0

    for batch_idx, start in enumerate(range(0, min(len(prompt_blocks), 10 * int(batch_size)), int(batch_size))):
        batch = prompt_blocks[start : start + int(batch_size)].to(device)
        total_generated_tokens += int(batch.shape[0]) * int(generated_len)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        tick = time.time()
        model.generate(
            input_ids=batch,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            do_sample=True,
            use_cache=True,
            top_k=50,
            max_length=int(original_len) + int(generated_len),
            top_p=0.95,
            temperature=1.0,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(device)))
        elapsed = time.time() - tick
        total_time += elapsed
        print(f"Batch {batch_idx + 1}: Time {elapsed:.2f} sec")

    if device.type == "cuda":
        total_memory_gb = peak_memory / (1024 ** 3)
        weight_memory_gb = weight_memory / (1024 ** 3)
        activation_memory_gb = max(0, peak_memory - weight_memory) / (1024 ** 3)
        print(f"Total Memory: {total_memory_gb:.2f} GB")
        print(f"Weight Memory: {weight_memory_gb:.2f} GB")
        print(f"Activation Memory: {activation_memory_gb:.2f} GB")

    if total_time > 0:
        print(f"Throughput: {total_generated_tokens / total_time:.2f} tokens/sec")
    else:
        print("Throughput could not be calculated.")
