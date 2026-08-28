from __future__ import annotations

import importlib.util
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

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
    clone_destination: str | None = None
    entrypoint: str | None = None
    notes: str = ""
    integration_status: str = "planned"
    supports_execute: bool = False
    required_packages: tuple[str, ...] = ()
    required_extra: tuple[str, ...] = ()


@dataclass(slots=True)
class BaselineHandle:
    spec: BaselineSpec
    path: str | None
    origin: str


@dataclass(slots=True)
class PreflightReport:
    family: str
    method: str
    integration_status: str
    supports_execute: bool
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    baseline_path: str | None = None


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
        integration_status="planned",
        required_packages=("torch", "transformers", "datasets", "accelerate"),
    ),
    ("svd", "aa_svd"): BaselineSpec(
        family="svd",
        method="aa_svd",
        display_name="AA-SVD",
        git_url="https://github.com/atulkumarin/AA-SVD.git",
        git_ref="1fa1b686cd9b13a77607a676564e37d438a176c8",
        local_candidates=("compress/svd/AA-SVD",),
        entrypoint="main.py",
        notes="Pinned LowRankArena source snapshot with a Hugging Face exporter.",
        integration_status="validated_recipe",
        supports_execute=True,
        required_packages=(
            "torch",
            "transformers",
            "datasets",
            "hydra",
            "omegaconf",
            "wandb",
        ),
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
    ("svd", "gfw_svd"): BaselineSpec(
        family="svd",
        method="gfw_svd",
        display_name="GFW-SVD",
        git_url="https://github.com/sayankotor/FisherKronecker.git",
        git_ref="d009b028c1e73545d8c604bcd29c1e091c8f341c",
        local_candidates=(
            "compress/external/svd/FisherKronecker",
            "compress/svd/GFW-SVD",
        ),
        clone_destination="compress/external/svd/FisherKronecker",
        entrypoint="llama/calibrate_llama_with_kronsvd.py",
        notes=(
            "LowRankArena adapter around the pinned FisherKronecker source. "
            "Execution requires precomputed Kronecker factors."
        ),
        integration_status="conditional",
        supports_execute=True,
        required_packages=(
            "torch",
            "transformers",
            "datasets",
            "numpy",
            "safetensors",
            "huggingface_hub",
        ),
        required_extra=("kron_factors_dir",),
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
    ("svd", "swift_svd"): BaselineSpec(
        family="svd",
        method="swift_svd",
        display_name="Swift-SVD",
        git_url="https://github.com/hiahei/Swift-SVD.git",
        git_ref="bd5be98340864deb5e51f120244ab43c446373d7",
        local_candidates=("compress/svd/Swift-SVD",),
        entrypoint="export_lowrank_hf.py",
        notes="Pinned LowRankArena source snapshot with uniform-rank HF export.",
        integration_status="validated_recipe",
        supports_execute=True,
        required_packages=(
            "torch",
            "transformers",
            "numpy",
            "safetensors",
            "huggingface_hub",
        ),
        required_extra=("svd_file",),
    ),
    ("svd", "zs_svd"): BaselineSpec(
        family="svd",
        method="zs_svd",
        display_name="ZS-SVD",
        git_url="https://github.com/mint-vu/Zero-Sum-SVD.git",
        git_ref="37e73f60875dbd5f0bf06327ae51d182c19fea33",
        local_candidates=("compress/svd/Zero-Sum-SVD",),
        entrypoint="main_zero_sum.py",
        notes="Pinned LowRankArena source snapshot with a Hugging Face exporter.",
        integration_status="validated_recipe",
        supports_execute=True,
        required_packages=(
            "torch",
            "transformers",
            "datasets",
            "accelerate",
            "safetensors",
        ),
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
    if spec.clone_destination:
        return project_path(spec.clone_destination)
    if spec.local_candidates:
        return project_path(spec.local_candidates[0])
    return COMPRESS_ROOT / spec.family / spec.display_name


def resolve_baseline(spec: BaselineSpec) -> BaselineHandle:
    for relative in spec.local_candidates:
        path = project_path(relative)
        if path.exists():
            origin = "git" if (path / ".git").exists() else "vendored"
            return BaselineHandle(spec=spec, path=str(path.resolve()), origin=origin)
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

    pinned_commit = re.fullmatch(r"[0-9a-fA-F]{40}", spec.git_ref) is not None

    if destination.exists() and (destination / ".git").exists():
        subprocess.run(
            ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", spec.git_ref],
            check=True,
        )
        if pinned_commit:
            subprocess.run(
                ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
                check=True,
            )
        else:
            # Preserve the historical branch-based refresh behaviour.
            subprocess.run(
                ["git", "-C", str(destination), "checkout", spec.git_ref],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(destination), "pull", "--ff-only", "origin", spec.git_ref],
                check=True,
            )
        return BaselineHandle(spec=spec, path=str(destination.resolve()), origin="git")

    if pinned_commit:
        subprocess.run(
            ["git", "clone", "--no-checkout", "--filter=blob:none", spec.git_url, str(destination)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", spec.git_ref],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
            check=True,
        )
    else:
        # Keep existing methods on the original shallow branch clone path.
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                spec.git_ref,
                spec.git_url,
                str(destination),
            ],
            check=True,
        )
    return BaselineHandle(spec=spec, path=str(destination.resolve()), origin="git")


def prepare_baseline(request: CompressionRequest) -> BaselineHandle:
    spec = get_baseline_spec(request.family, request.method)
    if request.clone_baseline or request.refresh_baseline:
        return clone_baseline(spec, request=request, refresh=request.refresh_baseline)
    return resolve_baseline(spec)


def _package_available(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def missing_packages(packages: Iterable[str]) -> list[str]:
    return [package for package in packages if not _package_available(package)]


def ensure_empty_output(output_dir: Path, unsafe_overwrite: bool) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not unsafe_overwrite:
            raise FileExistsError(f"Non-empty output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: Mapping[str, str] | None = None,
) -> None:
    env = os.environ.copy()
    if environment:
        env.update(environment)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}; see {log_path}"
        )


def preflight_request(
    request: CompressionRequest,
    baseline: BaselineHandle | None = None,
    *,
    for_execute: bool | None = None,
) -> PreflightReport:
    """Validate a compression request before model weights are loaded.

    This deliberately checks only deterministic local prerequisites. GPU memory,
    gated-model access, and the semantic validity of external calibration assets
    are checked by the method adapter at execution time.
    """

    spec = get_baseline_spec(request.family, request.method)
    baseline = baseline or prepare_baseline(request)
    execute = request.execute if for_execute is None else for_execute
    errors: list[str] = []
    warnings: list[str] = []

    if request.ratio is None:
        errors.append("--ratio is required")
    elif not 0.0 < request.ratio <= 1.0:
        errors.append("--ratio must be in (0, 1]")

    missing_packages = [
        package for package in spec.required_packages if not _package_available(package)
    ]
    if missing_packages:
        errors.append(
            "missing Python packages: " + ", ".join(sorted(missing_packages))
        )

    if baseline.path is None:
        message = (
            f"baseline source is unavailable; use --clone-baseline for {spec.git_url}"
            if spec.git_url
            else "baseline source is unavailable"
        )
        (errors if execute else warnings).append(message)
    elif spec.entrypoint and not (Path(baseline.path) / spec.entrypoint).exists():
        errors.append(
            f"baseline entrypoint is missing: {Path(baseline.path) / spec.entrypoint}"
        )

    if execute and not spec.supports_execute:
        errors.append(
            f"{request.family}/{request.method} is '{spec.integration_status}' and "
            "does not yet support --execute through the unified CLI"
        )

    if execute:
        for key in spec.required_extra:
            value = request.extra.get(key)
            if value in (None, ""):
                errors.append(f"missing required --extra {key}=...")
                continue
            if key.endswith(("_dir", "_file", "_path")):
                path = Path(str(value)).expanduser()
                if not path.exists():
                    errors.append(f"--extra {key} path does not exist: {path}")

    return PreflightReport(
        family=request.family,
        method=request.method,
        integration_status=spec.integration_status,
        supports_execute=spec.supports_execute,
        ok=not errors,
        errors=errors,
        warnings=warnings,
        baseline_path=baseline.path,
    )


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
