from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass


def configure_runtime_environment(*, verbose_vllm: bool) -> None:
    if not verbose_vllm:
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "CRITICAL")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@contextmanager
def use_safe_vllm_cwd():
    original_cwd = os.getcwd()
    safe_cwd = os.path.dirname(os.path.abspath(__file__))
    if original_cwd == safe_cwd:
        yield
        return
    os.chdir(safe_cwd)
    try:
        yield
    finally:
        os.chdir(original_cwd)


@dataclass
class ProgressPrinter:
    total_steps: int
    enabled: bool = True

    def step(self, index: int, message: str) -> None:
        if not self.enabled:
            return
        print(f"[{index}/{self.total_steps}] {message}", flush=True)

    def detail(self, message: str) -> None:
        if not self.enabled:
            return
        print(f"    {message}", flush=True)

    @contextmanager
    def waiting(self, index: int, message: str, *, interval_seconds: float = 5.0):
        if not self.enabled:
            yield
            return

        stop_event = threading.Event()
        started = time.time()

        def _worker() -> None:
            while not stop_event.wait(interval_seconds):
                elapsed = time.time() - started
                print(
                    f"[{index}/{self.total_steps}] {message} ({elapsed:.0f}s elapsed)",
                    flush=True,
                )

        print(f"[{index}/{self.total_steps}] {message}", flush=True)
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop_event.set()
            thread.join(timeout=interval_seconds)
            elapsed = time.time() - started
            print(
                f"[{index}/{self.total_steps}] {message} done ({elapsed:.2f}s)",
                flush=True,
            )


def print_json(data: str) -> None:
    sys.stdout.write(data)
    if not data.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


def print_failure(message: str) -> None:
    print(f"[error] {message}", flush=True)
