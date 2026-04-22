from __future__ import annotations

import gc
from dataclasses import dataclass, field

import torch


_GIB = float(1024 ** 3)
_MIB = int(1024 ** 2)


@dataclass(slots=True)
class CudaMemoryGuard:
    device: torch.device
    enabled: bool = False
    keep_free_gib: float = 60.0
    reserve_fraction: float = 0.95
    chunk_mib: int = 256
    min_chunk_mib: int = 16
    verbose: bool = True
    keep_free_bytes: int = field(init=False, default=0)
    chunk_bytes: int = field(init=False, default=0)
    min_chunk_bytes: int = field(init=False, default=0)
    _buffers: list[torch.Tensor] = field(init=False, default_factory=list)
    _reserved_bytes: int = field(init=False, default=0)

    def __post_init__(self):
        self.device = torch.device(self.device)
        self.enabled = bool(self.enabled and self.device.type == "cuda" and torch.cuda.is_available())
        self.keep_free_bytes = max(0, int(float(self.keep_free_gib) * _GIB))
        self.chunk_bytes = max(_MIB, int(self.chunk_mib) * _MIB)
        self.min_chunk_bytes = max(_MIB, int(self.min_chunk_mib) * _MIB)
        self._buffers: list[torch.Tensor] = []
        self._reserved_bytes = 0

    @property
    def reserved_bytes(self) -> int:
        return int(self._reserved_bytes)

    def _log(self, message: str):
        if self.enabled and self.verbose:
            print(f"[GpuGuard] {message}")

    def release_for_gpu_work(self, reason: str | None = None) -> int:
        if not self.enabled:
            return 0
        released = self._reserved_bytes
        with torch.cuda.device(self.device):
            if self._buffers:
                self._buffers.clear()
                self._reserved_bytes = 0
                gc.collect()
                torch.cuda.empty_cache()
                if reason:
                    self._log(f"released {released / _GIB:.2f} GiB for {reason}")
        return int(released)

    def reserve_idle(self, reason: str | None = None) -> int:
        if not self.enabled:
            return 0
        self.release_for_gpu_work()
        with torch.cuda.device(self.device):
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            target_bytes = max(0, int((free_bytes - self.keep_free_bytes) * float(self.reserve_fraction)))
            if target_bytes < self.min_chunk_bytes:
                return 0

            remaining = target_bytes
            next_chunk = min(self.chunk_bytes, remaining)
            while remaining >= self.min_chunk_bytes:
                try:
                    buf = torch.empty((next_chunk,), dtype=torch.uint8, device=self.device)
                    self._buffers.append(buf)
                    self._reserved_bytes += int(buf.numel())
                    remaining -= next_chunk
                    next_chunk = min(self.chunk_bytes, remaining)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    torch.cuda.empty_cache()
                    if next_chunk <= self.min_chunk_bytes:
                        break
                    next_chunk = max(self.min_chunk_bytes, next_chunk // 2)

        if reason and self._reserved_bytes > 0:
            self._log(
                f"reserved {self._reserved_bytes / _GIB:.2f} GiB after {reason} "
                f"(keep_free={self.keep_free_gib:.2f} GiB, fraction={self.reserve_fraction:.2f})"
            )
        return int(self._reserved_bytes)

    def close(self):
        self.release_for_gpu_work("guard shutdown")


def build_cuda_memory_guard(
    device: str | torch.device,
    *,
    enabled: bool = False,
    keep_free_gib: float = 60.0,
    reserve_fraction: float = 0.95,
    chunk_mib: int = 256,
    min_chunk_mib: int = 16,
    verbose: bool = True,
) -> CudaMemoryGuard:
    return CudaMemoryGuard(
        device=torch.device(device),
        enabled=enabled,
        keep_free_gib=keep_free_gib,
        reserve_fraction=reserve_fraction,
        chunk_mib=chunk_mib,
        min_chunk_mib=min_chunk_mib,
        verbose=verbose,
    )
