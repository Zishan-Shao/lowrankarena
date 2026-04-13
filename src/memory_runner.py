from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.inference_adapter import prepare_model_for_inference
from src.load import load_checkpoint
from src.result_schema import build_result_payload
from src.utils import dump_json, ensure_dir, run_results_root
from src.validation import validate_checkpoint_layout


DTYPE_MAP: dict[str, torch.dtype | str] = {
    "auto": "auto",
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@dataclass(slots=True)
class MemoryRequest:
    checkpoint_name: str
    index_path: str | Path
    output_dir: str | Path | None = None
    device: str = "cuda:0"
    dtype: str = "float16"
    batch_size: int = 1
    prompt_length: int = 32
    generation_length: int = 8
    attn_implementation: str | None = None
    trust_remote_code: bool = True
    local_files_only: bool = False
    verbose_backend: bool = False
    run_label: str = "ad_hoc"
    strict_validation: bool = False


@dataclass(slots=True)
class MemoryResult:
    checkpoint_name: str
    suite: str
    status: str
    output_path: str
    metrics: dict[str, Any]


def resolve_dtype(name: str) -> torch.dtype | str:
    key = name.strip().lower()
    if key not in DTYPE_MAP:
        supported = ", ".join(sorted(DTYPE_MAP))
        raise ValueError(f"Unsupported dtype {name!r}. Expected one of: {supported}")
    return DTYPE_MAP[key]


def bytes_to_gib(value: int) -> float:
    return value / float(1024**3)


def tensor_bytes(tensors: list[torch.Tensor]) -> int:
    return sum(int(t.numel() * t.element_size()) for t in tensors)


def snapshot_memory(device: torch.device) -> dict[str, int]:
    torch.cuda.synchronize(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def pick_filler_token_id(config: Any) -> int:
    vocab_size = getattr(config, "vocab_size", None)
    candidates = [
        getattr(config, "bos_token_id", None),
        getattr(config, "eos_token_id", None),
        getattr(config, "pad_token_id", None),
        1,
        0,
    ]
    for candidate in candidates:
        if isinstance(candidate, int) and candidate >= 0:
            if vocab_size is None or candidate < vocab_size:
                return candidate
    raise ValueError("Unable to find a valid synthetic token id for this checkpoint.")


def estimate_dense_kv_bytes(
    config: Any,
    *,
    batch_size: int,
    prompt_length: int,
    generation_length: int,
    bytes_per_elem: int,
) -> dict[str, int]:
    num_layers = int(getattr(config, "num_hidden_layers"))
    num_attention_heads = int(getattr(config, "num_attention_heads"))
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_attention_heads))
    hidden_size = int(getattr(config, "hidden_size"))
    head_dim = int(getattr(config, "head_dim", hidden_size // num_attention_heads))
    kv_bytes_per_token = num_layers * 2 * num_kv_heads * head_dim * bytes_per_elem
    cached_tokens_at_peak = batch_size * (prompt_length + max(generation_length - 1, 0))
    return {
        "bytes_per_token": kv_bytes_per_token,
        "cached_tokens_at_peak": cached_tokens_at_peak,
        "estimated_peak_kv_bytes": kv_bytes_per_token * cached_tokens_at_peak,
    }


def _configure_backend_logging(*, verbose_backend: bool) -> None:
    if verbose_backend:
        return
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    try:
        from transformers.utils import logging as transformers_logging
    except ImportError:
        return
    transformers_logging.set_verbosity_error()


def _load_model(
    model_path: str,
    *,
    dtype: torch.dtype | str,
    attn_implementation: str | None,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Any:
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def _result_path_for(request: MemoryRequest) -> Path:
    output_root = Path(request.output_dir) if request.output_dir else run_results_root("memory", request.run_label)
    ensure_dir(output_root)
    return output_root / f"memory__{request.checkpoint_name}.json"


def run_memory_measurement(request: MemoryRequest) -> MemoryResult:
    if request.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if request.prompt_length <= 0:
        raise ValueError("prompt_length must be positive.")
    if request.generation_length < 0:
        raise ValueError("generation_length cannot be negative.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU peak memory measurement.")

    device = torch.device(request.device)
    if device.type != "cuda":
        raise ValueError(f"device must be a CUDA device, got {request.device!r}")

    _configure_backend_logging(verbose_backend=request.verbose_backend)
    dtype = resolve_dtype(request.dtype)

    loaded = load_checkpoint(
        request.checkpoint_name,
        index_path=str(request.index_path),
        download=True,
        local_files_only=request.local_files_only,
        trust_remote_code=request.trust_remote_code,
    )
    prepared = prepare_model_for_inference(loaded)
    validation_summary = validate_checkpoint_layout(
        prepared.model_path,
        strict=request.strict_validation,
    )
    model_path = prepared.model_path
    if model_path.startswith("hf://"):
        raise RuntimeError("Expected a local checkpoint path after materialization, but only got an HF locator.")

    free_before, total_before = torch.cuda.mem_get_info(device)

    model = _load_model(
        model_path,
        dtype=dtype,
        attn_implementation=request.attn_implementation,
        trust_remote_code=request.trust_remote_code,
        local_files_only=True,
    )
    model.eval()
    model.to(device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    parameter_bytes = tensor_bytes(list(model.parameters()))
    buffer_bytes = tensor_bytes(list(model.buffers()))
    load_snapshot = snapshot_memory(device)

    config = model.config
    filler_token_id = pick_filler_token_id(config)
    input_ids = torch.full(
        (request.batch_size, request.prompt_length),
        fill_value=filler_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids, device=device)
    pad_token_id = getattr(config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(config, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = filler_token_id

    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=request.generation_length,
            min_new_tokens=request.generation_length,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
        )
    inference_snapshot = snapshot_memory(device)

    generated_tokens = int(generated.shape[1] - input_ids.shape[1])
    parameter_dtype = next(model.parameters()).dtype
    parameter_dtype_bytes = next(model.parameters()).element_size()
    kv_estimate = estimate_dense_kv_bytes(
        config,
        batch_size=request.batch_size,
        prompt_length=request.prompt_length,
        generation_length=generated_tokens,
        bytes_per_elem=parameter_dtype_bytes,
    )

    metrics = {
        "parameter_bytes": parameter_bytes,
        "parameter_gib": bytes_to_gib(parameter_bytes),
        "buffer_bytes": buffer_bytes,
        "buffer_gib": bytes_to_gib(buffer_bytes),
        "load_allocated_bytes": load_snapshot["allocated_bytes"],
        "load_allocated_gib": bytes_to_gib(load_snapshot["allocated_bytes"]),
        "load_reserved_bytes": load_snapshot["reserved_bytes"],
        "load_reserved_gib": bytes_to_gib(load_snapshot["reserved_bytes"]),
        "peak_allocated_bytes": inference_snapshot["max_allocated_bytes"],
        "peak_allocated_gib": bytes_to_gib(inference_snapshot["max_allocated_bytes"]),
        "peak_reserved_bytes": inference_snapshot["max_reserved_bytes"],
        "peak_reserved_gib": bytes_to_gib(inference_snapshot["max_reserved_bytes"]),
        "delta_peak_allocated_bytes_from_load": inference_snapshot["max_allocated_bytes"]
        - load_snapshot["allocated_bytes"],
        "delta_peak_allocated_gib_from_load": bytes_to_gib(
            inference_snapshot["max_allocated_bytes"] - load_snapshot["allocated_bytes"]
        ),
        "delta_peak_reserved_bytes_from_load": inference_snapshot["max_reserved_bytes"]
        - load_snapshot["reserved_bytes"],
        "delta_peak_reserved_gib_from_load": bytes_to_gib(
            inference_snapshot["max_reserved_bytes"] - load_snapshot["reserved_bytes"]
        ),
        "estimated_dense_kv_bytes_per_token": kv_estimate["bytes_per_token"],
        "estimated_dense_kv_bytes_per_token_gib": bytes_to_gib(kv_estimate["bytes_per_token"]),
        "estimated_peak_dense_kv_bytes": kv_estimate["estimated_peak_kv_bytes"],
        "estimated_peak_dense_kv_gib": bytes_to_gib(kv_estimate["estimated_peak_kv_bytes"]),
    }

    payload = build_result_payload(
        kind="memory",
        record=loaded.record,
        locator=loaded.locator,
        backend_name="transformers",
        backend_version=None,
        suite_path=None,
        suite_name="memory",
        config={
            "device": str(device),
            "dtype": str(parameter_dtype),
            "batch_size": request.batch_size,
            "prompt_length": request.prompt_length,
            "generation_length_requested": request.generation_length,
            "generation_length_actual": generated_tokens,
            "attn_implementation": request.attn_implementation,
            "trust_remote_code": request.trust_remote_code,
            "local_files_only": request.local_files_only,
        },
        metrics=metrics,
        artifacts={},
        runtime={
            "model_path": model_path,
            "tokenizer_path": prepared.tokenizer_path,
            "tokenizer_mode": prepared.tokenizer_mode,
            "preparation_kind": prepared.preparation_kind,
            "source_model_path": prepared.source_model_path,
            "preparation_notes": prepared.notes,
            "synthetic_input_token_id": filler_token_id,
            "device_memory_before_load": {
                "free_bytes": int(free_before),
                "total_bytes": int(total_before),
                "free_gib": bytes_to_gib(int(free_before)),
                "total_gib": bytes_to_gib(int(total_before)),
            },
        },
        validation=validation_summary,
        details={
            "load": {
                **load_snapshot,
                "allocated_gib": bytes_to_gib(load_snapshot["allocated_bytes"]),
                "reserved_gib": bytes_to_gib(load_snapshot["reserved_bytes"]),
            },
            "inference": {
                **inference_snapshot,
                "peak_allocated_gib": bytes_to_gib(inference_snapshot["max_allocated_bytes"]),
                "peak_reserved_gib": bytes_to_gib(inference_snapshot["max_reserved_bytes"]),
            },
            "estimated_dense_kv": {
                **kv_estimate,
                "bytes_per_token_gib": bytes_to_gib(kv_estimate["bytes_per_token"]),
                "estimated_peak_kv_gib": bytes_to_gib(kv_estimate["estimated_peak_kv_bytes"]),
            },
        },
        run_label=request.run_label,
        strict_validation=request.strict_validation,
    )
    output_path = dump_json(payload, _result_path_for(request))
    return MemoryResult(
        checkpoint_name=request.checkpoint_name,
        suite="memory",
        status="completed",
        output_path=str(output_path),
        metrics=metrics,
    )
