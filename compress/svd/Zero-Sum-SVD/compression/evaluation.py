"""
Evaluation utilities for SVD-LLM compression.

Contains helper functions for perplexity and commonsense evaluation.
"""

import sys
import io
import torch


def _patch_transformers_for_lm_eval():
    """
    Patch transformers module for compatibility with lm_eval.

    Adds missing cache classes that don't exist in transformers 4.35.2
    to avoid import errors when using newer lm_eval versions.
    """
    import transformers

    # cache classes that do not exist in transformers 4.35.2
    for name in ["Cache", "DynamicCache", "EncoderDecoderCache", "HybridCache"]:
        if not hasattr(transformers, name):
            setattr(transformers, name, type(name, (), {}))

    # optional, avoids other lm_eval wrappers crashing on older transformers
    if not hasattr(transformers, "Qwen2AudioForConditionalGeneration"):
        setattr(transformers, "Qwen2AudioForConditionalGeneration", type("Qwen2AudioForConditionalGeneration", (), {}))


@torch.no_grad()
def commonsense_eval(model, tokenizer, tasks_csv: str, device: str = "cuda"):
    """
    Run commonsense evaluation using lm_eval.

    Args:
        model: The language model
        tokenizer: Model tokenizer
        tasks_csv: Comma-separated list of lm_eval task names
        device: Device to use for evaluation

    Returns:
        Evaluation results from lm_eval
    """
    _patch_transformers_for_lm_eval()

    import logging
    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator

    # Suppress lm_eval warnings (e.g., "Failed to get model SHA")
    logging.getLogger("lm_eval").setLevel(logging.ERROR)

    model.to(device)
    model.eval()

    tasks = [t.strip() for t in tasks_csv.split(",") if t.strip()]
    if not tasks:
        raise ValueError("No commonsense tasks provided")

    # Suppress model print from HFLM initialization
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        hflm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=1,
            device=device,
        )
    finally:
        sys.stdout = old_stdout

    return evaluator.simple_evaluate(model=hflm, tasks=tasks)


def _sanitize_for_json(obj):
    """Recursively convert non-JSON-serializable objects."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, torch.dtype):
        return str(obj)
    elif hasattr(obj, 'item'):  # numpy/torch scalars
        return obj.item()
    return obj


def _extract_commonsense_metrics(cs_results):
    """Extract just acc and acc_norm from lm_eval results."""
    if cs_results is None:
        return None
    results = cs_results.get("results", {})
    simplified = {}
    for task, metrics in results.items():
        if isinstance(metrics, dict):
            simplified[task] = {
                "acc": metrics.get("acc,none"),
                "acc_norm": metrics.get("acc_norm,none"),
            }
    return simplified


def set_deterministic_seeds(seed: int):
    """
    Set seeds for reproducibility across different GPUs.

    Args:
        seed: Random seed to use
    """
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"🎲 Deterministic seeds set: {seed}")
