import logging
from typing import Dict

import torch
from torch import nn
from tqdm import tqdm
from datasets import load_dataset
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def evaluate_ppl(
    model: nn.Module,
    tokenizer,
    config: DictConfig,
) -> Dict[str, float]:
    """Evaluate perplexity on each dataset listed in config.datasets."""
    logger.info("Starting PPL evaluation")
    datasets = config.get("datasets", [])
    results = {}
    for dataset in datasets:
        logger.info("Evaluating PPL for dataset: %s", dataset)
        eval_loader = get_eval_loaders(dataset, tokenizer)
        perplexity = compute_ppl(
            dataset,
            model,
            tokenizer,
            eval_loader,
            config.get("use_bos", False),
            config.get("seqlen", 2048),
            config.get("batch_size", 1),
        )
        results[f'ppl_{dataset}'] = perplexity
        logger.info("PPL for %s: %f", dataset, perplexity)
    return results


@torch.no_grad()
def compute_ppl(
    dataset,
    model,
    tokenizer,
    eval_loader,
    use_bos,
    seqlen,
    batch_size=1,
    limit=-1,
):
    """
    Compute perplexity over the tokenized evaluation data in `eval_loader`.

    Slices the long token sequence into chunks of length `seqlen` and
    processes them in batches of `batch_size` sequences per forward pass.
    """
    testenc = eval_loader.input_ids
    if use_bos:
        # Prepend BOS to inputs but keep labels at length `seqlen`
        seqlen -= 1

    total_len = testenc.size(1)
    nsamples = total_len // seqlen
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()
    nlls = []
    loss_fct = nn.CrossEntropyLoss()
    processed = 0

    for start in tqdm(range(0, nsamples, batch_size)):
        end = min(start + batch_size, nsamples)
        curr_batch_size = end - start

        if limit != -1:
            if processed >= limit:
                break
            if processed + curr_batch_size > limit:
                end = start + (limit - processed)
                curr_batch_size = end - start

        # shape: (curr_batch_size, seqlen)
        labels = torch.stack(
            [
                testenc[0, i * seqlen:(i + 1) * seqlen]
                for i in range(start, end)
            ],
            dim=0,
        ).to(model.device)

        if use_bos:
            bos = torch.tensor(
                [[tokenizer.bos_token_id]] * labels.size(0)
            ).to(model.device)
            batch = torch.cat([bos, labels], dim=1)
        else:
            batch = labels

        hidden_states = model.model(batch)[0]
        if use_bos:
            # Drop BOS hidden state to align with labels
            hidden_states = hidden_states[:, 1:, :]

        logits = model.lm_head(hidden_states)
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:].to(model.device)

        # reshape handles potential non-contiguous tensors from slicing
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        nlls.append(loss.float().item())
        processed += curr_batch_size

    ppl = torch.exp(torch.tensor(sum(nlls)) / len(nlls)).item()
    logger.info("%s PPL: %.4f", dataset, ppl)
    model.config.use_cache = use_cache
    return ppl


def get_eval_loaders(name, tokenizer):
    """Load and tokenize a standard evaluation dataset by name."""
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="test"
        )
        return tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    if "ptb" in name:
        valdata = load_dataset(
            "ptb_text_only", "penn_treebank", split="test"
        )
        return tokenizer(
            "\n\n".join(valdata["sentence"]), return_tensors="pt"
        )

    if "c4" in name:
        url = (
            "https://huggingface.co/datasets/allenai/c4/resolve/main"
            "/en/c4-validation.00000-of-00008.json.gz"
        )
        testdata = load_dataset("json", data_files=url, split="train")
        return tokenizer(
            "\n\n".join(testdata["text"][:2000]), return_tensors="pt"
        )

    raise NotImplementedError(f"No eval loader for dataset: {name}")
