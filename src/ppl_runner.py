from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.benchmarking import suite_output_name
from src.dtype_utils import normalize_dtype_name
from src.inference_adapter import prepare_model_for_inference
from src.load import load_checkpoint
from src.memory_runner import resolve_dtype
from src.result_schema import build_result_payload
from src.utils import dump_json, ensure_dir, load_yaml, run_results_root
from src.validation import validate_checkpoint_layout

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional until eval dependencies are installed.
    load_dataset = None


DEFAULT_C4_STREAM_MAX_EVAL_TOKENS = 262144
DEFAULT_C4_DATASET_PATH = "allenai/c4"
DEFAULT_C4_DATASET_NAME = "en"
DEFAULT_WIKITEXT_DATASET_PATH = "wikitext"
DEFAULT_WIKITEXT_DATASET_NAME = "wikitext-2-raw-v1"


@dataclass(slots=True)
class ContiguousPplResult:
    checkpoint_name: str
    suite: str
    status: str
    output_path: str
    raw_output_path: str
    metrics: dict[str, Any]


def _result_path_for(request: Any, suite_path: Path) -> Path:
    output_root = Path(request.output_dir) if request.output_dir else run_results_root("eval", request.run_label)
    ensure_dir(output_root)
    return output_root / f"{suite_output_name(suite_path)}__{request.checkpoint_name}.json"


def _raw_output_path_for(request: Any, suite_path: Path) -> Path:
    raw_root = (
        Path(request.raw_output_root)
        if request.raw_output_root
        else run_results_root("eval", request.run_label) / "raw"
    )
    raw_dir = ensure_dir(raw_root / suite_output_name(suite_path) / request.checkpoint_name)
    return raw_dir / "contiguous_ppl.json"


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return [int(token_id) for token_id in encoded["input_ids"]]


def _build_contiguous_blocks(token_ids: list[int], *, max_length: int) -> torch.Tensor:
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1 for perplexity evaluation.")
    usable_tokens = (len(token_ids) // max_length) * max_length
    if usable_tokens < max_length:
        raise ValueError(
            f"Need at least {max_length} tokens to build one contiguous block, got {len(token_ids)}."
        )
    return torch.tensor(token_ids[:usable_tokens], dtype=torch.long).view(-1, max_length)


def _hash_token_ids(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, byteorder="little", signed=False))
    return digest.hexdigest()


def _load_wikitext2_token_ids(
    tokenizer: Any,
    *,
    split: str,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> list[int]:
    if load_dataset is None:
        raise RuntimeError("datasets is required for contiguous perplexity evaluation.")
    dataset = load_dataset(
        DEFAULT_WIKITEXT_DATASET_PATH,
        DEFAULT_WIKITEXT_DATASET_NAME,
        split=split,
        cache_dir=cache_dir,
        revision=revision,
    )
    return _encode_text(tokenizer, "\n\n".join(str(item) for item in dataset["text"]))


def _load_c4_stream_token_ids(
    tokenizer: Any,
    *,
    split: str,
    max_eval_tokens: int,
    max_length: int,
    cache_dir: str | None = None,
    revision: str | None = None,
    dataset_path: str = DEFAULT_C4_DATASET_PATH,
    dataset_name: str = DEFAULT_C4_DATASET_NAME,
    document_offset: int = 0,
) -> tuple[list[int], int]:
    if load_dataset is None:
        raise RuntimeError("datasets is required for contiguous perplexity evaluation.")
    target_tokens = (int(max_eval_tokens) // int(max_length)) * int(max_length)
    if target_tokens < max_length:
        raise ValueError(
            f"c4_stream max_eval_tokens must allow at least one full block of length {max_length}."
        )

    separator_tokens = _encode_text(tokenizer, "\n\n")
    docs_scanned = 0
    docs_skipped = 0
    token_ids: list[int] = []
    last_doc_had_content = False

    dataset_attempts = [(dataset_path, dataset_name)]
    if revision is None and dataset_path == DEFAULT_C4_DATASET_PATH and dataset_name == DEFAULT_C4_DATASET_NAME:
        dataset_attempts.append(("c4", DEFAULT_C4_DATASET_NAME))
    last_error: Exception | None = None
    stream = None
    for path, name in dataset_attempts:
        try:
            stream = load_dataset(
                path,
                name,
                split=split,
                streaming=True,
                cache_dir=cache_dir,
                revision=revision,
            )
            break
        except Exception as exc:  # pragma: no cover - depends on dataset availability.
            last_error = exc
    if stream is None:
        raise RuntimeError("Unable to load a C4 stream from Hugging Face.") from last_error

    for item in stream:
        text = str(item.get("text", "") or "")
        if not text.strip():
            continue
        if docs_skipped < document_offset:
            docs_skipped += 1
            continue
        doc_tokens = _encode_text(tokenizer, text)
        if not doc_tokens:
            continue
        if last_doc_had_content and separator_tokens:
            token_ids.extend(separator_tokens)
        token_ids.extend(doc_tokens)
        last_doc_had_content = True
        docs_scanned += 1
        if len(token_ids) >= target_tokens:
            break

    if len(token_ids) < target_tokens:
        raise ValueError(
            f"C4 stream did not yield enough tokens for the requested target ({target_tokens}); got {len(token_ids)}."
        )
    return token_ids[:target_tokens], docs_scanned


def _dataset_token_ids(
    dataset_config: dict[str, Any],
    *,
    tokenizer: Any,
    max_length: int,
    default_cache_dir: str | None,
) -> tuple[list[int], dict[str, Any]]:
    dataset_kind = str(dataset_config.get("kind") or dataset_config.get("name") or "").strip().lower()
    split = str(dataset_config.get("split", "test"))
    cache_dir = str(dataset_config.get("cache_dir", default_cache_dir)) if dataset_config.get("cache_dir", default_cache_dir) else None
    revision = str(dataset_config.get("revision")).strip() if dataset_config.get("revision") else None

    if dataset_kind == "wikitext2":
        token_ids = _load_wikitext2_token_ids(
            tokenizer,
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )
        return token_ids, {
            "split": split,
            "dataset_path": DEFAULT_WIKITEXT_DATASET_PATH,
            "dataset_config_name": DEFAULT_WIKITEXT_DATASET_NAME,
            "dataset_revision": revision,
        }

    if dataset_kind == "c4_stream":
        max_eval_tokens = int(dataset_config.get("max_eval_tokens", DEFAULT_C4_STREAM_MAX_EVAL_TOKENS))
        dataset_path = str(dataset_config.get("path") or dataset_config.get("dataset_path") or DEFAULT_C4_DATASET_PATH)
        dataset_name = str(dataset_config.get("config_name") or dataset_config.get("dataset_name") or DEFAULT_C4_DATASET_NAME)
        document_offset = int(dataset_config.get("document_offset", 0))
        token_ids, docs_scanned = _load_c4_stream_token_ids(
            tokenizer,
            split=split,
            max_eval_tokens=max_eval_tokens,
            max_length=max_length,
            cache_dir=cache_dir,
            revision=revision,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            document_offset=document_offset,
        )
        return token_ids, {
            "split": split,
            "dataset_path": dataset_path,
            "dataset_config_name": dataset_name,
            "dataset_revision": revision,
            "document_offset": document_offset,
            "max_eval_tokens": max_eval_tokens,
            "docs_scanned": docs_scanned,
        }

    raise ValueError(f"Unsupported contiguous_ppl dataset kind: {dataset_kind!r}")


def _evaluate_blocks(
    model: Any,
    blocks: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, int]:
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, int(blocks.shape[0]), int(batch_size)):
        batch = blocks[start : start + int(batch_size)].to(device)
        with torch.inference_mode():
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
    return total_nll, total_tokens


def _summarize_tasks(task_results: list[dict[str, Any]], *, aggregation: str) -> dict[str, Any]:
    ppls = {str(item["name"]): float(item["ppl"]) for item in task_results}
    if aggregation != "macro_mean":
        raise ValueError(f"Unsupported perplexity aggregation: {aggregation!r}")
    mean = sum(ppls.values()) / float(len(ppls)) if ppls else None
    return {
        "primary_metric": "ppl",
        "mean": mean,
        "task_count": len(task_results),
        "scored_task_count": len(task_results),
        "aggregation": aggregation,
        "tracked_metrics": ["ppl"],
        "by_metric": {"ppl": ppls},
    }


def run_contiguous_ppl_suite(request: Any, *, suite_path: Path, suite_config: dict[str, Any]) -> ContiguousPplResult:
    eval_config = dict(suite_config.get("eval") or {})
    dataset_configs = list(eval_config.get("datasets") or [])
    if not dataset_configs:
        raise ValueError(f"No contiguous_ppl datasets configured in {suite_path}.")

    batch_size = int(request.batch_size if request.batch_size is not None else eval_config.get("batch_size", 1))
    max_length = int(eval_config.get("max_length", 2048))
    aggregation = str(eval_config.get("metric_aggregation", "macro_mean"))
    dataset_cache_dir = eval_config.get("dataset_cache_dir")
    requested_device = request.device if request.device is not None else eval_config.get("device")
    device = torch.device(str(requested_device or ("cuda:0" if torch.cuda.is_available() else "cpu")))

    effective_dtype_name = normalize_dtype_name(
        request.extra_model_args.get("dtype", eval_config.get("dtype", "auto"))
    )
    dtype = resolve_dtype(effective_dtype_name)
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

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

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_kwargs = prepared.build_tokenizer_kwargs(trust_remote_code=request.trust_remote_code)
    tokenizer_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(prepared.tokenizer_path, **tokenizer_kwargs)

    model = AutoModelForCausalLM.from_pretrained(
        prepared.model_path,
        torch_dtype=dtype,
        trust_remote_code=request.trust_remote_code,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)

    task_results: list[dict[str, Any]] = []
    for dataset_config in dataset_configs:
        dataset_name = str(dataset_config.get("name") or dataset_config.get("kind") or "").strip()
        token_ids, dataset_meta = _dataset_token_ids(
            dataset_config,
            tokenizer=tokenizer,
            max_length=max_length,
            default_cache_dir=str(dataset_cache_dir) if dataset_cache_dir else None,
        )
        blocks = _build_contiguous_blocks(token_ids, max_length=max_length)
        total_nll, total_tokens = _evaluate_blocks(
            model,
            blocks,
            batch_size=batch_size,
            device=device,
        )
        scored_token_ids = blocks.reshape(-1).tolist()
        ppl = math.exp(total_nll / float(total_tokens))
        task_results.append(
            {
                "name": dataset_name,
                "ppl": ppl,
                "negative_log_likelihood_sum": total_nll,
                "token_count": total_tokens,
                "loaded_token_count": len(token_ids),
                "scored_token_sha256": _hash_token_ids(scored_token_ids),
                "block_count": int(blocks.shape[0]),
                "max_length": max_length,
                **dataset_meta,
            }
        )

    summary = _summarize_tasks(task_results, aggregation=aggregation)
    raw_output = {
        "backend": "contiguous_ppl",
        "version": str(eval_config.get("version", "1.0")),
        "tasks": task_results,
        "summary": summary,
    }
    raw_output_path = dump_json(raw_output, _raw_output_path_for(request, suite_path))

    payload = build_result_payload(
        kind="eval",
        record=loaded.record,
        locator=loaded.locator,
        backend_name=str(eval_config.get("backend", "contiguous_ppl")),
        backend_version=str(eval_config.get("version", "1.0")),
        suite_path=suite_path,
        suite_name=str(suite_config.get("name", suite_path.stem)),
        config={
            "datasets": [str(item.get("name") or item.get("kind") or "") for item in dataset_configs],
            "dataset_configs": dataset_configs,
            "metric": str(eval_config.get("metric", "ppl")),
            "metric_aggregation": aggregation,
            "dtype": str(dtype).replace("torch.", "") if dtype != "auto" else effective_dtype_name,
            "device": str(device),
            "batch_size": batch_size,
            "max_length": max_length,
            "tokenization": {
                "apply_chat_template": False,
                "add_special_tokens": False,
                "tokenizer_path": prepared.tokenizer_path,
            },
        },
        metrics=summary,
        artifacts={
            "raw_result_path": str(raw_output_path),
        },
        runtime={
            "model_path": prepared.model_path,
            "tokenizer_path": prepared.tokenizer_path,
            "tokenizer_mode": prepared.tokenizer_mode,
            "preparation_kind": prepared.preparation_kind,
            "source_model_path": prepared.source_model_path,
            "preparation_notes": prepared.notes,
        },
        validation=validation_summary,
        details={
            "summary": summary,
            "tasks": task_results,
        },
        run_label=request.run_label,
        strict_validation=request.strict_validation,
    )
    output_path = dump_json(payload, _result_path_for(request, suite_path))
    return ContiguousPplResult(
        checkpoint_name=request.checkpoint_name,
        suite=suite_output_name(suite_path),
        status="completed",
        output_path=str(output_path),
        raw_output_path=str(raw_output_path),
        metrics=summary,
    )
