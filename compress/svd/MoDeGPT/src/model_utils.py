import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaTokenizer


logger = logging.getLogger("MoDeGPT")


def _resolve_dtype_from_env(env_name: str, default: torch.dtype) -> torch.dtype:
    raw = os.environ.get(env_name, "").strip().lower()
    if raw in {"float32", "fp32", "f32"}:
        return torch.float32
    if raw in {"float64", "fp64", "f64", "double"}:
        return torch.float64
    if raw in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if raw in {"float16", "fp16", "f16", "half"}:
        return torch.float16
    return default


"""
precision to use for nearly all operations
 - float32 is enough for the 8B compression sweeps and keeps calibration memory sane.
 - Override MODEGPT_ACCUM_DTYPE explicitly if a future numerical audit needs double.
"""
dtype_p = _resolve_dtype_from_env("MODEGPT_ACCUM_DTYPE", torch.float32)
"""
The final type to cast all the weights BACK down to
"""
dtype_f = torch.float16

"""
True if we will be using multiple gpu's to compress
 - Inference/calibration will not be performed in parallel (though could be done)
 - Weights will be compressed on the 2nd gpu, while the first GPU holds the entire model
"""
parallel = False
conservative = True
d1 = "cuda:0"
d2 = "cuda:1" if parallel else "cuda:0"
# calib_device = "cpu"
calib_device = os.environ.get("MODEGPT_CALIB_DEVICE", "cuda:1" if parallel else "cuda:0")


def _load_tokenizer(tokenizer_source: str):
    def _valid(tokenizer):
        return tokenizer is not None and tokenizer is not False and hasattr(tokenizer, "pad_token") and callable(tokenizer)

    if os.path.exists(os.path.join(str(tokenizer_source), "tokenizer.model")):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
        if _valid(tokenizer):
            return tokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
        if _valid(tokenizer):
            return tokenizer
        logger.warning("Loaded invalid tokenizer %r from %s; retrying LlamaTokenizer.", tokenizer, tokenizer_source)
    except ValueError as exc:
        message = str(exc)
        if "Converting from SentencePiece and Tiktoken failed" not in message:
            raise
        logger.warning(
            "AutoTokenizer fast conversion failed for %s; retrying with LlamaTokenizer.",
            tokenizer_source,
        )
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_source, use_fast=False)
    if not _valid(tokenizer):
        raise TypeError(f"Loaded invalid tokenizer {type(tokenizer)!r} from {tokenizer_source}")
    return tokenizer


def start_memory_usage_worker():

    import psutil
    import time
    import threading

    def print_memory_usage():
        process = psutil.Process(os.getpid())
        mem_usage_path = os.environ.get("MODEGPT_MEM_USAGE_PATH", "./.mem-usage")
        mem_usage_dir = os.path.dirname(mem_usage_path)
        if mem_usage_dir:
            os.makedirs(mem_usage_dir, exist_ok=True)
        while True:
            mem_gb = process.memory_info().rss / (1024**3)
            sys_mem = psutil.virtual_memory()

            with open(mem_usage_path, "w") as f:
                f.write(
                    f"[Monitor] Process RAM: {mem_gb:.2f} GB\nSystem RAM: {sys_mem.percent}% used"
                )
                if mem_gb > 60:
                    f.write(
                        "\n\n⚠️ CRITICAL WARNING: Process nearing 64GB RAM limit! Crash imminent.\n"
                    )

            time.sleep(1)

    monitor_thread = threading.Thread(target=print_memory_usage, daemon=True)
    monitor_thread.start()

    return monitor_thread


def load_model(model_name: str, device: int = 0):
    """
    Loads the official model.
    """
    logger.info(f"Loading model from: {model_name}")
    tokenizer = _load_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True, torch_dtype="auto"
    )

    logger.info(f"params.dtype = {next(model.parameters()).dtype}")
    logger.info(f"model.config.dtype = {model.config.torch_dtype}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("No pad_token found. Set pad_token = eos_token.")

    return model, tokenizer, model.config


def save_compressed_model(
    adapter,
    rotary_masks: torch.Tensor | None,
    save_dir: str,
    source_model_name: str,
):
    import shutil

    model, tokenizer = adapter.model, adapter.tokenizer

    arch = adapter.arch
    if arch == "opt":
        rebuild_path = "./src/patchers/OPTRebuild.py"
    elif arch == "llama":
        rebuild_path = "./src/patchers/LlamaRebuild.py"
    elif "qwen" in arch:
        rebuild_path = "./src/patchers/DenseQwenRebuild.py"
    else:
        raise Exception("Cannot save compressed model ... no compressed model definition")

    os.makedirs(save_dir, exist_ok=True)

    if rotary_masks is not None:
        mask_path = os.path.abspath(os.path.join(save_dir, "rotary_masks.pt"))
        model.config.mask_path = mask_path
    else:
        model.config.mask_path = None

    model.config.torch_dtype = "bfloat16"
    model.config.dtype = "bfloat16"

    logger.info(f"params.dtype = {next(model.parameters()).dtype}")
    logger.info(f"model.config.dtype = {model.config.torch_dtype}")

    logger.info(f"Saving compressed model to {save_dir}")
    model.save_pretrained(save_dir, safe_serialization=False)
    tokenizer.save_pretrained(save_dir)
    if rotary_masks is not None:
        torch.save(rotary_masks, mask_path)
    shutil.copy(rebuild_path, save_dir)
    with open(os.path.join(save_dir, "tokenizer_source.txt"), "w") as f:
        f.write(source_model_name.strip())

    logger.info(f"✔ Model, tokenizer, and tokenizer_source.txt saved to {save_dir}")


def reload_compressed_model(model_dir: str, device="cuda:0", tokenizer_source: str = ""):
    """
    Better just to use this always. As long as you pass tokenizer_source (which is just the name of the model)
    it will always work compressed or not.
    """
    logger.info(f"Reloading compressed model from: {model_dir}")
    if not tokenizer_source:
        tokenizer_source_path = os.path.join(model_dir, "tokenizer_source.txt")

        if os.path.exists(tokenizer_source_path):
            with open(tokenizer_source_path, "r") as f:
                tokenizer_source = f.read().strip()
        else:
            tokenizer_source = model_dir

    tokenizer = _load_tokenizer(tokenizer_source)
    # from transformers import LlamaTokenizer

    # tokenizer = LlamaTokenizer.from_pretrained(tokenizer_source)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",  # low_cpu_mem_usage=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("No pad_token found. Set pad_token = eos_token.")

    model.to(device)
    logger.info(f"✔ Loaded model on cuda:{device} with float16.")
    logger.info(f"✔ Reloaded compressed model to {device} successfully.")

    model.eval()
    return model, tokenizer
