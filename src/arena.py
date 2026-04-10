from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from src.benchmarking import BENCHMARK_ROOT, resolve_suite_path
from src.lm_eval_runner import LmEvalRequest, run_lm_eval_suite
from src.registry import (
    DEFAULT_INDEX_PATH,
    CheckpointRecord,
    merge_checkpoint_indexes,
    save_checkpoint_index,
    upsert_checkpoint,
)
from src.report import build_table
from src.speed_runner import VllmSpeedRequest, run_vllm_speed_suite
from src.utils import dump_json, ensure_dir, load_json, project_path


DEFAULT_MANIFEST_ROOT = project_path("checkpoints", "manifests")


def _manifest_id(payload: dict[str, Any]) -> str:
    checkpoint_id = str(payload.get("id") or payload.get("name") or "").strip()
    if not checkpoint_id:
        raise ValueError("Manifest must define a non-empty 'id' or 'name'.")
    return checkpoint_id


def _manifest_path(root: str | Path, checkpoint_id: str) -> Path:
    return Path(root) / f"{checkpoint_id}.json"


def _record_dict(record: CheckpointRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["id"] = record.name
    return payload


class Arena:
    """Thin facade over the existing registry, eval, speed, and report modules."""

    def __init__(
        self,
        registries: list[str | Path] | None = None,
        benchmark_root: str | Path = BENCHMARK_ROOT,
        manifests_root: str | Path = DEFAULT_MANIFEST_ROOT,
    ) -> None:
        resolved_registries = [Path(path) for path in (registries or [DEFAULT_INDEX_PATH])]
        self.registries = resolved_registries
        self.benchmark_root = Path(benchmark_root)
        self.manifests_root = Path(manifests_root)
        self._overlay_records: dict[str, CheckpointRecord] = {}
        self._overlay_manifests: dict[str, dict[str, Any]] = {}
        self._manifest_cache = self._load_manifest_cache()

    def _load_manifest_cache(self) -> dict[str, dict[str, Any]]:
        if not self.manifests_root.exists():
            return {}
        manifests: dict[str, dict[str, Any]] = {}
        for path in sorted(self.manifests_root.rglob("*.json")):
            payload = load_json(path)
            checkpoint_id = _manifest_id(payload)
            payload.setdefault("manifest_path", str(path.resolve()))
            manifests[checkpoint_id] = payload
        return manifests

    def _manifest_for(self, checkpoint_id: str) -> dict[str, Any] | None:
        if checkpoint_id in self._overlay_manifests:
            return self._overlay_manifests[checkpoint_id]
        if checkpoint_id in self._manifest_cache:
            return self._manifest_cache[checkpoint_id]

        candidate = _manifest_path(self.manifests_root, checkpoint_id)
        if not candidate.exists():
            return None

        payload = load_json(candidate)
        payload.setdefault("manifest_path", str(candidate.resolve()))
        self._manifest_cache[checkpoint_id] = payload
        return payload

    def _merged_records(self) -> dict[str, CheckpointRecord]:
        merged = {record.name: record for record in merge_checkpoint_indexes(self.registries)}
        merged.update(self._overlay_records)
        return merged

    def _record_for(self, checkpoint_id: str) -> CheckpointRecord:
        records = self._merged_records()
        if checkpoint_id not in records:
            known = ", ".join(sorted(records))
            raise KeyError(f"Unknown checkpoint '{checkpoint_id}'. Known checkpoints: {known}")
        return records[checkpoint_id]

    def _merged_description(self, record: CheckpointRecord) -> dict[str, Any]:
        payload = _record_dict(record)
        manifest = self._manifest_for(record.name)
        if manifest:
            for key, value in manifest.items():
                if key in {"id", "name"}:
                    continue
                payload[key] = value
        return payload

    @contextmanager
    def _runtime_index(self) -> Iterator[Path]:
        if len(self.registries) == 1 and not self._overlay_records:
            yield self.registries[0]
            return

        with tempfile.NamedTemporaryFile(prefix="lra-index-", suffix=".csv", delete=False) as handle:
            runtime_index = Path(handle.name)

        try:
            save_checkpoint_index(self._merged_records().values(), path=runtime_index)
            yield runtime_index
        finally:
            runtime_index.unlink(missing_ok=True)

    def list(
        self,
        *,
        benchmark: str | None = None,
        method: str | None = None,
        enabled_only: bool = True,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for record in sorted(self._merged_records().values(), key=lambda item: item.name.lower()):
            if enabled_only and not record.enabled:
                continue
            if benchmark and benchmark not in record.benchmarks:
                continue
            if method and record.method != method:
                continue
            rows.append(self._merged_description(record))
        return rows

    def describe(self, checkpoint_id: str) -> dict[str, Any]:
        return self._merged_description(self._record_for(checkpoint_id))

    def register(
        self,
        *,
        id: str,
        source: str = "huggingface",
        repo_id: str = "",
        revision: str = "main",
        subpath: str = "",
        path: str | Path | None = None,
        model_family: str = "",
        variant: str = "unknown",
        method: str = "pending",
        benchmarks: list[str] | None = None,
        enabled: bool = True,
        notes: str = "",
        family: str | None = None,
        base_model: str | None = None,
        ratio: float | None = None,
        precision: str | None = None,
        calibration: str | None = None,
        recovery: str | None = None,
        format: str | None = None,
        eval_compatible: bool | None = None,
        speed_compatible: bool | None = None,
        manifest_path: str | Path | None = None,
        persist: bool = False,
        index_path: str | Path | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        resolved_subpath = str(path) if path is not None else subpath
        if source == "huggingface" and not repo_id:
            raise ValueError("repo_id is required when source='huggingface'.")
        if source == "local" and not resolved_subpath:
            raise ValueError("path or subpath is required when source='local'.")

        record = CheckpointRecord(
            name=id,
            model_family=model_family,
            variant=variant,
            method=method,
            source=source,
            repo_id=repo_id,
            revision=revision or "main",
            subpath=resolved_subpath,
            benchmarks=list(benchmarks or []),
            enabled=enabled,
            notes=notes,
        )
        self._overlay_records[id] = record

        manifest = {
            "schema_version": "1.0",
            "id": id,
            "source": source,
            "repo_id": repo_id,
            "revision": revision or "main",
            "subpath": resolved_subpath,
            "model_family": model_family,
            "variant": variant,
            "method": method,
            "benchmarks": list(benchmarks or []),
            "enabled": enabled,
            "notes": notes,
        }
        optional_fields = {
            "family": family,
            "base_model": base_model,
            "ratio": ratio,
            "precision": precision,
            "calibration": calibration,
            "recovery": recovery,
            "format": format,
            "eval_compatible": eval_compatible,
            "speed_compatible": speed_compatible,
        }
        for key, value in optional_fields.items():
            if value is not None and value != "":
                manifest[key] = value
        for key, value in extra.items():
            if value is not None:
                manifest[key] = value

        self._overlay_manifests[id] = manifest
        if persist:
            self._persist_manifest(manifest, manifest_path=manifest_path)
            target_index = Path(index_path) if index_path else self.registries[0]
            upsert_checkpoint(record, path=target_index)

        return self.describe(id)

    def register_manifest(
        self,
        manifest_path: str | Path,
        *,
        persist: bool = False,
        index_path: str | Path | None = None,
    ) -> dict[str, Any]:
        source_manifest_path = Path(manifest_path).resolve()
        manifest = load_json(source_manifest_path)
        checkpoint_id = _manifest_id(manifest)
        return self.register(
            id=checkpoint_id,
            source=str(manifest.get("source", "huggingface")),
            repo_id=str(manifest.get("repo_id", "")),
            revision=str(manifest.get("revision", "main") or "main"),
            subpath=str(manifest.get("subpath") or manifest.get("path") or ""),
            model_family=str(manifest.get("model_family") or manifest.get("model") or ""),
            variant=str(manifest.get("variant", "unknown")),
            method=str(manifest.get("method", "pending")),
            benchmarks=[str(item) for item in manifest.get("benchmarks", [])],
            enabled=bool(manifest.get("enabled", True)),
            notes=str(manifest.get("notes", "")),
            family=manifest.get("family"),
            base_model=manifest.get("base_model"),
            ratio=manifest.get("ratio"),
            precision=manifest.get("precision"),
            calibration=manifest.get("calibration"),
            recovery=manifest.get("recovery"),
            format=manifest.get("format"),
            eval_compatible=manifest.get("eval_compatible"),
            speed_compatible=manifest.get("speed_compatible"),
            manifest_path=None,
            persist=persist,
            index_path=index_path,
            source_manifest=str(source_manifest_path),
            upstream_config=manifest.get("upstream_config"),
            loader_note=manifest.get("loader_note"),
            recommended_loader=manifest.get("recommended_loader"),
            checkpoint_file=manifest.get("checkpoint_file"),
            timing_file=manifest.get("timing_file"),
            stage=manifest.get("stage"),
            whitening_nsamples=manifest.get("whitening_nsamples"),
        )

    def evaluate(
        self,
        checkpoint_id: str,
        *,
        suite: str | Path = "accuracy/mcq",
        output_dir: str | Path | None = None,
        raw_output_root: str | Path | None = None,
        lm_eval_bin: str | None = None,
        model_backend: str = "hf",
        device: str | None = None,
        batch_size: str | int | None = None,
        limit: float | int | None = None,
        num_fewshot: int | None = None,
        log_samples: bool = False,
        use_cache: str | None = None,
        trust_remote_code: bool = True,
        extra_model_args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manifest = self._manifest_for(checkpoint_id) or {}
        if manifest.get("eval_compatible") is False:
            raise ValueError(f"Checkpoint '{checkpoint_id}' is registered as eval-incompatible.")

        suite_path = resolve_suite_path(suite, root=self.benchmark_root)
        with self._runtime_index() as index_path:
            result = run_lm_eval_suite(
                LmEvalRequest(
                    checkpoint_name=checkpoint_id,
                    suite_path=suite_path,
                    index_path=index_path,
                    output_dir=output_dir,
                    raw_output_root=raw_output_root,
                    lm_eval_bin=lm_eval_bin or "lm-eval",
                    model_backend=model_backend,
                    device=device,
                    batch_size=batch_size,
                    limit=limit,
                    num_fewshot=num_fewshot,
                    log_samples=log_samples,
                    use_cache=use_cache,
                    trust_remote_code=trust_remote_code,
                    extra_model_args=dict(extra_model_args or {}),
                )
            )
        return asdict(result)

    def speed(
        self,
        checkpoint_id: str,
        *,
        suite: str | Path = "speed/speed",
        output_dir: str | Path | None = None,
        batch_sizes: list[int] | None = None,
        prompt_lengths: list[int] | None = None,
        generation_lengths: list[int] | None = None,
        repeat: int | None = None,
        warmup: int | None = None,
        tensor_parallel_size: int | None = None,
        gpu_memory_utilization: float | None = None,
        dtype: str | None = None,
        enforce_eager: bool | None = None,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
    ) -> dict[str, Any]:
        manifest = self._manifest_for(checkpoint_id) or {}
        if manifest.get("speed_compatible") is False:
            raise ValueError(f"Checkpoint '{checkpoint_id}' is registered as speed-incompatible.")

        suite_path = resolve_suite_path(suite, root=self.benchmark_root)
        with self._runtime_index() as index_path:
            result = run_vllm_speed_suite(
                VllmSpeedRequest(
                    checkpoint_name=checkpoint_id,
                    suite_path=suite_path,
                    index_path=index_path,
                    output_dir=output_dir,
                    batch_sizes=batch_sizes,
                    prompt_lengths=prompt_lengths,
                    generation_lengths=generation_lengths,
                    repeat=repeat,
                    warmup=warmup,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    dtype=dtype,
                    enforce_eager=enforce_eager,
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only,
                )
            )
        return asdict(result)

    def report(
        self,
        *,
        kind: str,
        columns: list[str] | None = None,
        result_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        filename: str | None = None,
    ) -> str:
        return str(
            build_table(
                kind=kind,
                columns=columns,
                result_dir=result_dir,
                output_dir=output_dir,
                filename=filename,
            )
        )

    def _persist_manifest(
        self,
        manifest: dict[str, Any],
        *,
        manifest_path: str | Path | None = None,
    ) -> Path:
        target = Path(manifest_path) if manifest_path else _manifest_path(self.manifests_root, _manifest_id(manifest))
        ensure_dir(target.parent)
        payload = dict(manifest)
        payload["manifest_path"] = str(target.resolve())
        dump_json(payload, target)
        self._manifest_cache[_manifest_id(payload)] = payload
        return target
