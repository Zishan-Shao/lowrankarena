from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit.common import (
    PROJECT_ROOT,
    ci95,
    dump_json,
    ensure_dir,
    eval_result_path,
    format_template,
    load_json,
    load_yaml,
    mean,
    model_slug,
    numeric_metric_from_eval_payload,
    ratio_tag,
    sample_std,
    shell_join,
    slugify,
    utc_timestamp,
)


OOM_PATTERNS = (
    "out of memory",
    "cuda out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "torch.cuda.outofmemoryerror",
)


def _python_bin(config: dict[str, Any]) -> str:
    return str(config.get("python", "python"))


def _append_option(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    command.extend([flag, str(value)])


def _context_values(
    *,
    config: dict[str, Any],
    method: str | None = None,
    calibration_profile: str | None = None,
    seed: int | None = None,
    subset: str | None = None,
) -> dict[str, Any]:
    compression = config.get("compression", {})
    model = str(compression.get("model", "model"))
    ratio = compression.get("ratio")
    return {
        "audit_id": config.get("audit_id"),
        "method": method,
        "model": model,
        "model_slug": model_slug(model),
        "ratio": ratio,
        "ratio_tag": ratio_tag(ratio),
        "calibration_profile": calibration_profile,
        "seed": seed,
        "subset": subset,
    }


def _eval_command(
    *,
    python: str,
    checkpoint: str,
    suite: str,
    run_label: str,
    output_dir: str | None,
    raw_output_dir: str | None,
    options: dict[str, Any],
) -> list[str]:
    command = [python, "scripts/run_eval.py", checkpoint, "--suite", suite, "--run-label", run_label]
    _append_option(command, "--output-dir", output_dir)
    _append_option(command, "--raw-output-dir", raw_output_dir)
    _append_option(command, "--model-backend", options.get("model_backend"))
    _append_option(command, "--device", options.get("device"))
    _append_option(command, "--dtype", options.get("dtype"))
    _append_option(command, "--batch-size", options.get("batch_size"))
    _append_option(command, "--limit", options.get("limit"))
    _append_option(command, "--num-fewshot", options.get("num_fewshot"))
    _append_option(command, "--tensor-parallel-size", options.get("tensor_parallel_size"))
    _append_option(command, "--gpu-memory-utilization", options.get("gpu_memory_utilization"))
    _append_option(command, "--max-model-len", options.get("max_model_len"))
    _append_option(command, "--enforce-eager", options.get("enforce_eager"))
    for item in options.get("model_args", []) or []:
        command.extend(["--model-arg", str(item)])
    return command


def _compression_command(
    *,
    python: str,
    config: dict[str, Any],
    method: str,
    calibration: str,
    seed: int,
    output_root: str,
    extra: dict[str, Any],
) -> list[str]:
    compression = config.get("compression", {})
    command = [
        python,
        "scripts/run_compress.py",
        "--family",
        str(compression.get("family", "svd")),
        "--method",
        method,
        "--model",
        str(compression["model"]),
        "--calibration",
        calibration,
        "--seed",
        str(seed),
        "--output-root",
        output_root,
    ]
    _append_option(command, "--ratio", compression.get("ratio"))
    _append_option(command, "--tokenizer", compression.get("tokenizer"))
    _append_option(command, "--revision", compression.get("revision"))
    _append_option(command, "--precision", compression.get("precision"))
    _append_option(command, "--recovery", compression.get("recovery"))
    _append_option(command, "--source", compression.get("source"))
    _append_option(command, "--notes", compression.get("notes"))
    _append_option(command, "--baseline-root", compression.get("baseline_root"))
    _append_option(command, "--clone-baseline", compression.get("clone_baseline"))
    _append_option(command, "--refresh-baseline", compression.get("refresh_baseline"))
    _append_option(command, "--execute", compression.get("execute"))
    _append_option(command, "--register", compression.get("register"))
    _append_option(command, "--enabled", compression.get("enabled"))
    merged_extra = dict(compression.get("extra") or {})
    merged_extra.update(extra)
    for key, value in sorted(merged_extra.items()):
        command.extend(["--extra", f"{key}={value}"])
    return command


def _command_entry(name: str, stage: str, command: list[str], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "stage": stage,
        "command": command,
        "shell": shell_join(command),
        "metadata": metadata,
    }


def _enabled_items(items: list[dict[str, Any]], *, include_disabled: bool = False) -> list[dict[str, Any]]:
    if include_disabled:
        return list(items)
    return [item for item in items if item.get("enabled", True)]


def _build_eval_matrix_plan(config: dict[str, Any], *, include_disabled: bool = False) -> list[dict[str, Any]]:
    python = _python_bin(config)
    audit_id = str(config["audit_id"])
    run_label = str(config.get("run_label", f"appendix_{audit_id.lower()}"))
    eval_config = config.get("eval", {})
    options = dict(eval_config.get("options") or {})
    output_dir = eval_config.get("output_dir")
    raw_output_dir = eval_config.get("raw_output_dir")
    commands: list[dict[str, Any]] = []
    for checkpoint_item in _enabled_items(list(eval_config.get("checkpoints", [])), include_disabled=include_disabled):
        checkpoint = str(checkpoint_item["checkpoint"])
        for suite in eval_config.get("suites", []):
            name = f"{audit_id}_{slugify(checkpoint)}_{slugify(suite)}"
            command = _eval_command(
                python=python,
                checkpoint=checkpoint,
                suite=str(suite),
                run_label=run_label,
                output_dir=output_dir,
                raw_output_dir=raw_output_dir,
                options=options,
            )
            commands.append(
                _command_entry(
                    name,
                    "eval",
                    command,
                    {
                        "audit_id": audit_id,
                        "checkpoint": checkpoint,
                        "suite": str(suite),
                        "label": checkpoint_item.get("label"),
                    },
                )
            )
    return commands


def _build_compression_eval_plan(config: dict[str, Any], *, include_disabled: bool = False) -> list[dict[str, Any]]:
    python = _python_bin(config)
    audit_id = str(config["audit_id"])
    run_label = str(config.get("run_label", f"appendix_{audit_id.lower()}"))
    compression = config.get("compression", {})
    evaluation = config.get("evaluation", {})
    methods = [str(item) for item in compression.get("methods", [])]
    checkpoint_template = str(evaluation["checkpoint_template"])
    eval_options = dict(evaluation.get("options") or {})
    eval_output_dir = evaluation.get("output_dir")
    raw_output_dir = evaluation.get("raw_output_dir")
    commands: list[dict[str, Any]] = []

    if config.get("kind") == "calibration_audit":
        profiles = _enabled_items(list(compression.get("calibration_profiles", [])), include_disabled=include_disabled)
        for method in methods:
            for profile in profiles:
                profile_name = str(profile["name"])
                context = _context_values(config=config, method=method, calibration_profile=profile_name)
                output_root_template = str(
                    compression.get("output_root_template", compression.get("output_root", "results/audit/artifacts"))
                )
                output_root = format_template(output_root_template, context)
                extra = {"audit_id": audit_id, "calibration_profile": profile_name}
                command = _compression_command(
                    python=python,
                    config=config,
                    method=method,
                    calibration=str(profile["calibration"]),
                    seed=int(profile.get("seed", compression.get("seed", 0))),
                    output_root=output_root,
                    extra=extra,
                )
                commands.append(
                    _command_entry(
                        f"{audit_id}_{slugify(method)}_{slugify(profile_name)}_compress",
                        "compress",
                        command,
                        {**context, "calibration": profile["calibration"]},
                    )
                )
                checkpoint = format_template(checkpoint_template, context)
                for suite in evaluation.get("suites", []):
                    eval_command = _eval_command(
                        python=python,
                        checkpoint=checkpoint,
                        suite=str(suite),
                        run_label=run_label,
                        output_dir=eval_output_dir,
                        raw_output_dir=raw_output_dir,
                        options=eval_options,
                    )
                    commands.append(
                        _command_entry(
                            f"{audit_id}_{slugify(method)}_{slugify(profile_name)}_{slugify(suite)}",
                            "eval",
                            eval_command,
                            {**context, "checkpoint": checkpoint, "suite": str(suite)},
                        )
                    )

    elif config.get("kind") == "stability":
        subsets = _enabled_items(list(compression.get("calibration_subsets", [])), include_disabled=include_disabled)
        for method in methods:
            for subset in subsets:
                subset_name = str(subset["name"])
                seed = int(subset.get("seed", compression.get("seed", 0)))
                context = _context_values(config=config, method=method, seed=seed, subset=subset_name)
                output_root_template = str(
                    compression.get("output_root_template", compression.get("output_root", "results/audit/artifacts"))
                )
                output_root = format_template(output_root_template, context)
                extra = {"audit_id": audit_id, "calibration_subset": subset_name}
                if subset.get("offset") is not None:
                    extra["calibration_offset"] = subset["offset"]
                command = _compression_command(
                    python=python,
                    config=config,
                    method=method,
                    calibration=str(subset.get("calibration", compression.get("calibration", "wikitext2"))),
                    seed=seed,
                    output_root=output_root,
                    extra=extra,
                )
                commands.append(
                    _command_entry(
                        f"{audit_id}_{slugify(method)}_{slugify(subset_name)}_compress",
                        "compress",
                        command,
                        {**context, "calibration": subset.get("calibration"), "offset": subset.get("offset")},
                    )
                )
                checkpoint = format_template(checkpoint_template, context)
                for suite in evaluation.get("suites", []):
                    eval_command = _eval_command(
                        python=python,
                        checkpoint=checkpoint,
                        suite=str(suite),
                        run_label=run_label,
                        output_dir=eval_output_dir,
                        raw_output_dir=raw_output_dir,
                        options=eval_options,
                    )
                    commands.append(
                        _command_entry(
                            f"{audit_id}_{slugify(method)}_{slugify(subset_name)}_{slugify(suite)}",
                            "eval",
                            eval_command,
                            {**context, "checkpoint": checkpoint, "suite": str(suite)},
                        )
                    )
    else:
        raise ValueError(f"Unsupported compression/eval audit kind: {config.get('kind')}")

    return commands


def _build_feasibility_plan(config: dict[str, Any], *, include_disabled: bool = False) -> list[dict[str, Any]]:
    python = _python_bin(config)
    audit_id = str(config["audit_id"])
    feasibility = config.get("feasibility", {})
    output_dir = str(feasibility.get("output_dir", f"results/audit/{audit_id.lower()}_feasibility"))
    command = [
        python,
        "audit/run_audit.py",
        "run-feasibility",
        str(config.get("_config_path", "")),
        "--output-dir",
        output_dir,
    ]
    if include_disabled:
        command.append("--include-disabled")
    return [
        _command_entry(
            f"{audit_id}_run_feasibility",
            "feasibility",
            command,
            {"audit_id": audit_id, "output_dir": output_dir},
        )
    ]


def build_plan(config_path: str | Path, *, include_disabled: bool = False) -> dict[str, Any]:
    config = load_yaml(config_path)
    config["_config_path"] = str(Path(config_path))
    kind = str(config.get("kind", ""))
    if kind == "eval_matrix":
        commands = _build_eval_matrix_plan(config, include_disabled=include_disabled)
    elif kind in {"calibration_audit", "stability"}:
        commands = _build_compression_eval_plan(config, include_disabled=include_disabled)
    elif kind == "feasibility":
        commands = _build_feasibility_plan(config, include_disabled=include_disabled)
    else:
        raise ValueError(f"Unsupported audit kind in {config_path}: {kind!r}")

    return {
        "schema_version": "audit_plan_v1",
        "generated_at": utc_timestamp(),
        "config_path": str(config_path),
        "audit_id": config.get("audit_id"),
        "kind": kind,
        "title": config.get("title"),
        "command_count": len(commands),
        "commands": commands,
    }


def write_shell_script(plan: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shell_join([str(PROJECT_ROOT)])}",
        "",
    ]
    for command in plan["commands"]:
        lines.append(f"echo '[{command['stage']}] {command['name']}'")
        lines.append(command["shell"])
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    output_path.chmod(0o755)
    return output_path


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _classify_failure(stdout: str, stderr: str, returncode: int) -> str:
    combined = f"{stdout}\n{stderr}".lower()
    if any(pattern in combined for pattern in OOM_PATTERNS):
        return "oom"
    if returncode == 124:
        return "timeout"
    return "error"


def run_feasibility(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    include_disabled: bool = False,
    only: set[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    audit_id = str(config["audit_id"])
    run_label = str(config.get("run_label", f"appendix_{audit_id.lower()}"))
    python = _python_bin(config)
    feasibility = config.get("feasibility", {})
    defaults = dict(feasibility.get("defaults") or {})
    root = ensure_dir(output_dir or feasibility.get("output_dir", f"results/audit/{audit_id.lower()}_feasibility"))
    log_root = ensure_dir(root / "logs")
    result_root = ensure_dir(root / "results")
    targets = _enabled_items(list(feasibility.get("targets", [])), include_disabled=include_disabled)

    records: list[dict[str, Any]] = []
    for target in targets:
        checkpoint = str(target["checkpoint"])
        if only and checkpoint not in only and str(target.get("name", "")) not in only:
            continue
        target_name = str(target.get("name", checkpoint))
        suite = str(target.get("suite", defaults.get("suite", "memory/active")))
        command = [
            python,
            "scripts/run_memory.py",
            checkpoint,
            "--suite",
            suite,
            "--output-dir",
            str(root / "memory"),
            "--run-label",
            run_label,
        ]
        for option_name, flag in (
            ("device", "--device"),
            ("dtype", "--dtype"),
            ("batch_size", "--batch-size"),
            ("prompt_length", "--prompt-length"),
            ("generation_length", "--generation-length"),
            ("attn_implementation", "--attn-implementation"),
        ):
            value = target.get(option_name, defaults.get(option_name))
            _append_option(command, flag, value)
        if bool(target.get("strict_validation", defaults.get("strict_validation", False))):
            command.append("--strict-validation")

        stdout_path = log_root / f"{slugify(target_name)}.stdout.log"
        stderr_path = log_root / f"{slugify(target_name)}.stderr.log"
        started = time.perf_counter()
        if dry_run:
            completed = None
            stdout = ""
            stderr = ""
            returncode = None
            status = "planned"
        else:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=target.get("timeout_seconds", defaults.get("timeout_seconds")),
            )
            stdout = completed.stdout
            stderr = completed.stderr
            returncode = int(completed.returncode)
            stdout_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")
            status = "completed" if returncode == 0 else f"failed_{_classify_failure(stdout, stderr, returncode)}"
        elapsed = time.perf_counter() - started

        parsed_stdout = _extract_json_object(stdout) if not dry_run else None
        memory_payload = None
        if parsed_stdout and parsed_stdout.get("output_path"):
            output_path = Path(str(parsed_stdout["output_path"]))
            if output_path.exists():
                memory_payload = load_json(output_path)

        record = {
            "schema_version": "audit_feasibility_v1",
            "generated_at": utc_timestamp(),
            "audit_id": audit_id,
            "status": status,
            "target": target,
            "checkpoint": checkpoint,
            "command": command,
            "shell": shell_join(command),
            "returncode": returncode,
            "wall_clock_seconds": elapsed,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "memory_result_path": parsed_stdout.get("output_path") if parsed_stdout else None,
            "peak_allocated_gib": (
                memory_payload.get("metrics", {}).get("peak_allocated_gib") if memory_payload else None
            ),
            "peak_reserved_gib": (
                memory_payload.get("metrics", {}).get("peak_reserved_gib") if memory_payload else None
            ),
            "cuda_runtime": memory_payload.get("runtime", {}).get("cuda_runtime") if memory_payload else None,
        }
        dump_json(record, result_root / f"feasibility__{slugify(target_name)}.json")
        records.append(record)

    summary = {
        "schema_version": "audit_feasibility_summary_v1",
        "generated_at": utc_timestamp(),
        "audit_id": audit_id,
        "config_path": str(config_path),
        "result_count": len(records),
        "results": records,
    }
    dump_json(summary, root / "summary.json")
    write_feasibility_markdown(summary, root / "table.md")
    return summary


def write_feasibility_markdown(summary: dict[str, Any], path: str | Path) -> Path:
    rows = [
        "| Checkpoint | Method | Ratio | Status | Peak Alloc GiB | Wall Clock s |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for item in summary.get("results", []):
        target = item.get("target", {})
        rows.append(
            "| {checkpoint} | {method} | {ratio} | {status} | {peak} | {wall} |".format(
                checkpoint=item.get("checkpoint"),
                method=target.get("method", ""),
                ratio=target.get("ratio", ""),
                status=item.get("status"),
                peak="" if item.get("peak_allocated_gib") is None else f"{item['peak_allocated_gib']:.3f}",
                wall="" if item.get("wall_clock_seconds") is None else f"{item['wall_clock_seconds']:.1f}",
            )
        )
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output_path


def summarize_calibration(config_path: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    config = load_yaml(config_path)
    compression = config.get("compression", {})
    evaluation = config.get("evaluation", {})
    eval_output_root = Path(evaluation.get("output_dir", "results/eval"))
    summary_root = ensure_dir(output_dir or "results/audit/4_calibration")
    checkpoint_template = str(evaluation["checkpoint_template"])
    profiles = [profile for profile in compression.get("calibration_profiles", []) if profile.get("enabled", True)]
    methods = [str(method) for method in compression.get("methods", [])]
    suites = [str(suite) for suite in evaluation.get("suites", [])]
    profile_names = [str(profile["name"]) for profile in profiles]
    baseline_profile = str(evaluation.get("baseline_profile", profile_names[0] if profile_names else ""))
    comparison_profile = str(evaluation.get("comparison_profile", profile_names[-1] if profile_names else ""))
    rows = []
    for method in methods:
        for suite in suites:
            scores: dict[str, float | None] = {}
            paths: dict[str, str] = {}
            for profile_name in profile_names:
                context = _context_values(config=config, method=method, calibration_profile=profile_name)
                checkpoint = format_template(checkpoint_template, context)
                path = eval_result_path(eval_output_root, suite, checkpoint)
                paths[profile_name] = str(path)
                scores[profile_name] = numeric_metric_from_eval_payload(load_json(path)) if path.exists() else None
            baseline = scores.get(baseline_profile)
            comparison = scores.get(comparison_profile)
            rows.append(
                {
                    "method": method,
                    "suite": suite,
                    "scores": scores,
                    "delta": None if baseline is None or comparison is None else comparison - baseline,
                    "result_paths": paths,
                }
            )
    summary = {
        "schema_version": "audit_calibration_summary_v1",
        "generated_at": utc_timestamp(),
        "audit_id": config.get("audit_id"),
        "eval_output_root": str(eval_output_root),
        "baseline_profile": baseline_profile,
        "comparison_profile": comparison_profile,
        "rows": rows,
    }
    dump_json(summary, summary_root / "summary.json")
    write_calibration_markdown(summary, summary_root / "delta_table.md")
    return summary


def summarize_stability(config_path: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    config = load_yaml(config_path)
    compression = config.get("compression", {})
    evaluation = config.get("evaluation", {})
    eval_output_root = Path(evaluation.get("output_dir", "results/eval"))
    summary_root = ensure_dir(output_dir or "results/audit/3_stability")
    checkpoint_template = str(evaluation["checkpoint_template"])
    subsets = [subset for subset in compression.get("calibration_subsets", []) if subset.get("enabled", True)]
    methods = [str(method) for method in compression.get("methods", [])]
    suites = [str(suite) for suite in evaluation.get("suites", [])]
    rows = []
    for method in methods:
        for suite in suites:
            values: list[float] = []
            observations = []
            for subset in subsets:
                subset_name = str(subset["name"])
                seed = int(subset.get("seed", compression.get("seed", 0)))
                context = _context_values(config=config, method=method, seed=seed, subset=subset_name)
                checkpoint = format_template(checkpoint_template, context)
                path = eval_result_path(eval_output_root, suite, checkpoint)
                score = numeric_metric_from_eval_payload(load_json(path)) if path.exists() else None
                if score is not None:
                    values.append(score)
                observations.append(
                    {
                        "subset": subset_name,
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "score": score,
                        "result_path": str(path),
                    }
                )
            rows.append(
                {
                    "method": method,
                    "suite": suite,
                    "n": len(values),
                    "mean": mean(values),
                    "std": sample_std(values),
                    "ci95": ci95(values),
                    "observations": observations,
                }
            )
    summary = {
        "schema_version": "audit_stability_summary_v1",
        "generated_at": utc_timestamp(),
        "audit_id": config.get("audit_id"),
        "eval_output_root": str(eval_output_root),
        "rows": rows,
    }
    dump_json(summary, summary_root / "summary.json")
    write_stability_markdown(summary, summary_root / "stability_table.md")
    return summary


def write_calibration_markdown(summary: dict[str, Any], path: str | Path) -> Path:
    baseline = str(summary.get("baseline_profile"))
    comparison = str(summary.get("comparison_profile"))
    rows = [
        f"| Method | Suite | {baseline} | {comparison} | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in summary.get("rows", []):
        scores = item.get("scores", {})
        baseline_value = scores.get(baseline)
        comparison_value = scores.get(comparison)
        delta = item.get("delta")
        rows.append(
            "| {method} | {suite} | {base} | {comp} | {delta} |".format(
                method=item.get("method"),
                suite=item.get("suite"),
                base="" if baseline_value is None else f"{baseline_value:.6f}",
                comp="" if comparison_value is None else f"{comparison_value:.6f}",
                delta="" if delta is None else f"{delta:.6f}",
            )
        )
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output_path


def write_stability_markdown(summary: dict[str, Any], path: str | Path) -> Path:
    rows = [
        "| Method | Suite | N | Mean | Std | CI95 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summary.get("rows", []):
        rows.append(
            "| {method} | {suite} | {n} | {mean} | {std} | {ci95} |".format(
                method=item.get("method"),
                suite=item.get("suite"),
                n=item.get("n"),
                mean="" if item.get("mean") is None else f"{item['mean']:.6f}",
                std="" if item.get("std") is None else f"{item['std']:.6f}",
                ci95="" if item.get("ci95") is None else f"{item['ci95']:.6f}",
            )
        )
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output_path


def run_plan(plan_path: str | Path, *, output_dir: str | Path | None = None, only_stage: str | None = None) -> dict[str, Any]:
    plan = load_json(plan_path)
    root = ensure_dir(output_dir or Path(plan_path).with_suffix("").with_name(Path(plan_path).stem + "_run"))
    log_root = ensure_dir(root / "logs")
    results = []
    for command in plan.get("commands", []):
        if only_stage and command.get("stage") != only_stage:
            continue
        name = str(command["name"])
        stdout_path = log_root / f"{slugify(name)}.stdout.log"
        stderr_path = log_root / f"{slugify(name)}.stderr.log"
        started = time.perf_counter()
        completed = subprocess.run(
            [str(item) for item in command["command"]],
            cwd=str(PROJECT_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        results.append(
            {
                "name": name,
                "stage": command.get("stage"),
                "returncode": int(completed.returncode),
                "wall_clock_seconds": time.perf_counter() - started,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
        )
        if completed.returncode != 0:
            break
    summary = {
        "schema_version": "audit_run_plan_summary_v1",
        "generated_at": utc_timestamp(),
        "plan_path": str(plan_path),
        "results": results,
    }
    dump_json(summary, root / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan and run LowRankArena secondary audits.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Build a command plan from an audit YAML config.")
    plan.add_argument("config")
    plan.add_argument("--output", default=None)
    plan.add_argument("--script", default=None)
    plan.add_argument("--include-disabled", action="store_true")

    run_plan_parser = subparsers.add_parser("run-plan", help="Execute commands from an audit plan JSON.")
    run_plan_parser.add_argument("plan")
    run_plan_parser.add_argument("--output-dir", default=None)
    run_plan_parser.add_argument("--only-stage", default=None)

    feasibility = subparsers.add_parser("run-feasibility", help="Run priority-1 feasibility probes.")
    feasibility.add_argument("config")
    feasibility.add_argument("--output-dir", default=None)
    feasibility.add_argument("--include-disabled", action="store_true")
    feasibility.add_argument("--only", action="append", default=[])
    feasibility.add_argument("--dry-run", action="store_true")

    summarize_cal = subparsers.add_parser("summarize-calibration", help="Summarize priority-4 calibration deltas.")
    summarize_cal.add_argument("config")
    summarize_cal.add_argument("--output-dir", default=None)

    summarize_stab = subparsers.add_parser("summarize-stability", help="Summarize priority-3 stability std/CI.")
    summarize_stab.add_argument("config")
    summarize_stab.add_argument("--output-dir", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        plan = build_plan(args.config, include_disabled=args.include_disabled)
        if args.output:
            dump_json(plan, args.output)
        if args.script:
            write_shell_script(plan, args.script)
        print(json.dumps(plan, indent=2, sort_keys=True))
    elif args.command == "run-plan":
        summary = run_plan(args.plan, output_dir=args.output_dir, only_stage=args.only_stage)
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "run-feasibility":
        summary = run_feasibility(
            args.config,
            output_dir=args.output_dir,
            include_disabled=args.include_disabled,
            only=set(args.only) if args.only else None,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "summarize-calibration":
        summary = summarize_calibration(args.config, output_dir=args.output_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "summarize-stability":
        summary = summarize_stability(args.config, output_dir=args.output_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
