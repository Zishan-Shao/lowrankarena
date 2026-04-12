from __future__ import annotations

import importlib.machinery
import importlib.metadata
import importlib.util
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent


def _filtered_sys_path() -> list[str]:
    filtered: list[str] = []
    repo_root = REPO_ROOT.resolve()
    for entry in sys.path:
        resolved = Path(entry or ".").resolve()
        if resolved == repo_root:
            continue
        filtered.append(entry)
    return filtered


def import_installed_vllm():
    existing = sys.modules.get("vllm")
    if existing is not None:
        module_file = getattr(existing, "__file__", "") or ""
        if module_file and not str(Path(module_file).resolve()).startswith(str(REPO_ROOT)):
            return existing

    spec = importlib.machinery.PathFinder.find_spec("vllm", _filtered_sys_path())
    if spec is None or spec.loader is None:
        raise ImportError(
            "Could not resolve the installed 'vllm' package without the local project path. "
            "Make sure the conda environment has vLLM installed."
        )

    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("vllm")
    sys.modules["vllm"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = previous
        raise
    return module


def installed_vllm_version() -> str | None:
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return None
