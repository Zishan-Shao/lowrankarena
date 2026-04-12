from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.memory_runner import MemoryRequest as MemoryRunnerRequest
from src.memory_runner import MemoryResult, run_memory_measurement


@dataclass(slots=True)
class MemoryRequest:
    checkpoint_name: str
    output_dir: str | Path | None = None
    device: str = "cuda:0"
    batch_size: int = 1
    prompt_length: int = 32
    generation_length: int = 8
    extra: dict[str, Any] = field(default_factory=dict)


def run_memory(request: MemoryRequest, index_path: str | None = None) -> MemoryResult:
    if index_path is None:
        raise ValueError("index_path is required for memory runs.")

    return run_memory_measurement(
        MemoryRunnerRequest(
            checkpoint_name=request.checkpoint_name,
            index_path=index_path,
            output_dir=request.output_dir,
            device=request.device,
            batch_size=request.batch_size,
            prompt_length=request.prompt_length,
            generation_length=request.generation_length,
            dtype=request.extra.get("dtype", "float16"),
            attn_implementation=request.extra.get("attn_implementation"),
            local_files_only=bool(request.extra.get("local_files_only", False)),
            verbose_backend=bool(request.extra.get("verbose_backend", False)),
        )
    )
