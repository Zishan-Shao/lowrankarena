from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils import ensure_dir, project_path, utc_timestamp


COMPRESS_ROOT = project_path("compress")
DEFAULT_ARTIFACT_ROOT = COMPRESS_ROOT / "artifacts"


@dataclass(slots=True)
class BaselineSpec:
    family: str
    method: str
    display_name: str
    git_url: str | None = None
    git_ref: str = "main"
    local_candidates: tuple[str, ...] = ()
    entrypoint: str | None = None
    notes: str = ""


@dataclass(slots=True)
class BaselineHandle:
    spec: BaselineSpec
    path: str | None
    origin: str


@dataclass(slots=True)
class CompressionRequest:
    family: str
    method: str
    model: str
    ratio: float | None = None
    output_root: str | Path | None = None
    tokenizer: str | None = None
    revision: str = "main"
    precision: str = "bf16"
    calibration: str = "wikitext2"
    recovery: str = "default"
    seed: int = 0
    source: str = "huggingface"
    notes: str = ""
    clone_baseline: bool = False
    refresh_baseline: bool = False
    execute: bool = False
    register: bool = False
    enabled: bool = False
    baseline_root: str | Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def artifact_root(self) -> Path:
        return Path(self.output_root) if self.output_root else DEFAULT_ARTIFACT_ROOT

    @property
    def model_slug(self) -> str:
        token = self.model.split("/")[-1]
        return _slugify(token)

    @property
    def ratio_tag(self) -> str:
        if self.ratio is None:
            return "na"
        ratio_value = self.ratio * 100 if self.ratio <= 1 else self.ratio
        return f"r{int(round(ratio_value)):02d}"

    @property
    def artifact_id(self) -> str:
        return f"{self.model_slug}_{_slugify(self.method)}_{self.ratio_tag}"

    def to_manifest_fields(self) -> dict[str, Any]:
        return {
            "id": self.artifact_id,
            "model": self.model,
            "method": self.method,
            "family": self.family,
            "ratio": self.ratio,
            "precision": self.precision,
            "calibration": self.calibration,
            "recovery": self.recovery,
            "source": "local",
            "dense_source": self.source,
            "dense_revision": self.revision,
            "tokenizer": self.tokenizer or self.model,
            "notes": self.notes,
            "extra": self.extra,
            "created_at": utc_timestamp(),
        }


BASELINE_SPECS: dict[tuple[str, str], BaselineSpec] = {
    ("svd", "asvd"): BaselineSpec(
        family="svd",
        method="asvd",
        display_name="ASVD",
        git_url="https://github.com/hahnyuan/ASVD4LLM.git",
        local_candidates=("compress/svd/ASVD",),
        entrypoint="asvd.py",
        notes="Activation-aware SVD baseline.",
    ),
    ("svd", "dobi_svd"): BaselineSpec(
        family="svd",
        method="dobi_svd",
        display_name="Dobi-SVD",
        local_candidates=("compress/svd/Dobi-SVD",),
        entrypoint="svd_trainer.py",
        notes="Vendored local snapshot is available; no official clone URL is configured yet.",
    ),
    ("svd", "fwsvd"): BaselineSpec(
        family="svd",
        method="fwsvd",
        display_name="FWSVD",
        local_candidates=("compress/svd/FWSVD",),
        notes="Expected to be implemented from local project code rather than a third-party repo.",
    ),
    ("svd", "svd"): BaselineSpec(
        family="svd",
        method="svd",
        display_name="SVD",
        local_candidates=("compress/svd/SVD",),
        notes="Plain SVD baseline is expected to live in-tree.",
    ),
    ("svd", "svd_llm"): BaselineSpec(
        family="svd",
        method="svd_llm",
        display_name="SVD-LLM",
        local_candidates=("compress/svd/SVD-LLM",),
        entrypoint="SVDLLM.py",
        notes="Vendored local snapshot is available.",
    ),
    ("svd", "basis_sharing"): BaselineSpec(
        family="svd",
        method="basis_sharing",
        display_name="basis-sharing",
        notes="Basis sharing is expected to be integrated as a local method wrapper.",
    ),
    ("prune", "slicegpt"): BaselineSpec(
        family="prune",
        method="slicegpt",
        display_name="SliceGPT",
        git_url="https://github.com/microsoft/TransformerCompression.git",
        local_candidates=("compress/prune/SliceGPT",),
        notes="SliceGPT lives inside Microsoft's TransformerCompression repo.",
    ),
    ("prune", "llm_pruner"): BaselineSpec(
        family="prune",
        method="llm_pruner",
        display_name="LLM-Pruner",
        git_url="https://github.com/horseee/LLM-Pruner.git",
        local_candidates=("compress/prune/LLM-Pruner",),
        notes="Official NeurIPS 2023 LLM-Pruner implementation.",
    ),
    ("prune", "wanda_sp"): BaselineSpec(
        family="prune",
        method="wanda_sp",
        display_name="Wanda",
        git_url="https://github.com/locuslab/wanda.git",
        local_candidates=("compress/prune/Wanda",),
        entrypoint="main.py",
        notes="Official Wanda pruning repo.",
    ),
    ("prune", "bonsai"): BaselineSpec(
        family="prune",
        method="bonsai",
        display_name="Bonsai",
        git_url="https://github.com/ldery/Bonsai.git",
        local_candidates=("compress/prune/Bonsai",),
        entrypoint="main.py",
        notes="Official Bonsai repo; note that it saves masks rather than a full HF export by default.",
    ),
    ("quant", "awq"): BaselineSpec(
        family="quant",
        method="awq",
        display_name="llm-awq",
        git_url="https://github.com/mit-han-lab/llm-awq.git",
        local_candidates=("compress/quant/llm-awq",),
        notes="Official AWQ implementation from MIT Han Lab.",
    ),
    ("quant", "gptq"): BaselineSpec(
        family="quant",
        method="gptq",
        display_name="AutoGPTQ",
        git_url="https://github.com/AutoGPTQ/AutoGPTQ.git",
        local_candidates=("compress/quant/AutoGPTQ",),
        notes="AutoGPTQ is used here as the practical GPTQ baseline implementation.",
    ),
    ("quant", "rtn"): BaselineSpec(
        family="quant",
        method="rtn",
        display_name="rtn",
        notes="RTN can be implemented with in-tree transformers/bitsandbytes tooling; no external repo is required.",
    ),
}


def _slugify(value: str) -> str:
    normalized = value.replace("/", "__")
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", normalized)
    normalized = normalized.strip("-_.").lower()
    return normalized or "artifact"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def get_baseline_spec(family: str, method: str) -> BaselineSpec:
    key = (family, method)
    if key not in BASELINE_SPECS:
        raise KeyError(f"Unsupported compression method: {family}/{method}")
    return BASELINE_SPECS[key]


def baseline_destination(spec: BaselineSpec, request: CompressionRequest | None = None) -> Path:
    if request and request.baseline_root:
        return Path(request.baseline_root) / spec.family / spec.display_name
    if spec.local_candidates:
        return project_path(spec.local_candidates[0])
    return COMPRESS_ROOT / spec.family / spec.display_name


def resolve_baseline(spec: BaselineSpec) -> BaselineHandle:
    for relative in spec.local_candidates:
        path = project_path(relative)
        if path.exists():
            return BaselineHandle(spec=spec, path=str(path.resolve()), origin="vendored")
    return BaselineHandle(spec=spec, path=None, origin="missing")


def clone_baseline(
    spec: BaselineSpec,
    request: CompressionRequest | None = None,
    refresh: bool = False,
) -> BaselineHandle:
    if not spec.git_url:
        return resolve_baseline(spec)

    destination = baseline_destination(spec, request=request)
    ensure_dir(destination.parent)
    if destination.exists() and not refresh:
        origin = "git" if (destination / ".git").exists() else "vendored"
        return BaselineHandle(spec=spec, path=str(destination.resolve()), origin=origin)

    if destination.exists() and not (destination / ".git").exists():
        return BaselineHandle(spec=spec, path=str(destination.resolve()), origin="vendored")

    if destination.exists() and (destination / ".git").exists():
        subprocess.run(
            ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", spec.git_ref],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", spec.git_ref],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "pull", "--ff-only", "origin", spec.git_ref],
            check=True,
        )
        return BaselineHandle(spec=spec, path=str(destination.resolve()), origin="git")

    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", spec.git_ref, spec.git_url, str(destination)],
        check=True,
    )
    return BaselineHandle(spec=spec, path=str(destination.resolve()), origin="git")


def prepare_baseline(request: CompressionRequest) -> BaselineHandle:
    spec = get_baseline_spec(request.family, request.method)
    if request.clone_baseline or request.refresh_baseline:
        return clone_baseline(spec, request=request, refresh=request.refresh_baseline)
    return resolve_baseline(spec)


def build_method_command(
    request: CompressionRequest,
    baseline: BaselineHandle,
    relative_output_dir: str,
) -> list[str] | None:
    ratio_text = str(request.ratio) if request.ratio is not None else ""
    if request.family == "svd" and request.method == "asvd":
        return [
            "python",
            "asvd.py",
            f"--model_id={request.model}",
            "--act_aware",
            "--alpha",
            "0.5",
            "--n_calib_samples",
            "32",
            "--scaling_method",
            "abs_mean",
            "--param_ratio_target",
            ratio_text,
            "--use_cache",
        ]
    if request.family == "svd" and request.method == "svd_llm":
        return [
            "python",
            "SVDLLM.py",
            "--model",
            request.model,
            "--step",
            "1",
            "--ratio",
            ratio_text,
            "--dataset",
            request.calibration,
            "--seed",
            str(request.seed),
            "--save_path",
            relative_output_dir,
        ]
    if request.family == "prune" and request.method == "wanda_sp":
        return [
            "python",
            "main.py",
            "--model",
            request.model,
            "--sparsity_ratio",
            ratio_text,
            "--save",
            relative_output_dir,
        ]
    if request.family == "prune" and request.method == "bonsai":
        return [
            "python",
            "main.py",
            "--model",
            request.model,
            "--dataset",
            request.calibration,
            "--sparsity_ratio",
            ratio_text,
            "--save",
            relative_output_dir,
        ]
    return None
