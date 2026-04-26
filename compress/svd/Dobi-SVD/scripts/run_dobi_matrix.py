#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run Dobi-SVD matrix experiments with OOM-aware GPU fallback.")
    parser.add_argument("--repo-root", type=str, default=".", help="Path to Dobi-SVD repo root")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python executable for jobs")
    parser.add_argument("--path-head-folder", type=str, required=True, help="path_head_folder for data/model cache")
    parser.add_argument("--path-head-folder-output", type=str, required=True, help="path_head_folder_output for results")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "/deac/csc/yangGrp/cuij/LLM/models/hf_models/huggyllama__llama-7b",
            "/deac/csc/yangGrp/cuij/LLM/models/hf_models/Qwen__Qwen3-8B-Base",
            "/deac/csc/yangGrp/cuij/LLM/models/hf_models/Qwen__Qwen3-8B",
        ],
        help="Model IDs/paths",
    )
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.4, 0.5, 0.6, 0.7, 0.8], help="Target ratios")
    parser.add_argument(
        "--gpu-tries",
        nargs="+",
        default=["0", "0,1", "0,1,2", "0,1,2,3"],
        help="GPU sets to try for trainer, in order",
    )
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--training-dataset", type=str, default="wikitext2")
    parser.add_argument("--n-train-epochs", type=int, default=20)
    parser.add_argument("--n-train-samples", type=int, default=256)
    parser.add_argument("--n-eval-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--remapping", action="store_true")
    parser.add_argument("--owner", type=str, default=os.environ.get("USER", "owner"))
    parser.add_argument("--method-name", type=str, default="DobiSVD")
    parser.add_argument("--output-format", type=str, default="pt")
    parser.add_argument(
        "--max-gamma",
        type=int,
        default=0,
        help="Optional env DOBI_MAX_GAMMA for smoke/low-memory run. 0 disables.",
    )
    parser.add_argument("--run-name", type=str, default="", help="Custom run folder name")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def conda_nvidia_lib_path():
    prefix = os.environ.get("CONDA_PREFIX", "")
    if not prefix:
        return ""
    libs = [
        "cublas",
        "cusparse",
        "cusolver",
        "cuda_runtime",
        "cuda_nvrtc",
        "nvjitlink",
    ]
    parts = [f"{prefix}/lib/python3.10/site-packages/nvidia/{x}/lib" for x in libs]
    return ":".join(parts)


def run_monitored(repo_root, python_bin, monitor_script, stage, gpu_set, log_file, metrics_file, poll_seconds, env_extra, cmd):
    env_json = json.dumps(env_extra) if env_extra else ""
    full_cmd = [
        python_bin,
        monitor_script,
        "--gpu-ids",
        gpu_set,
        "--log-file",
        str(log_file),
        "--metrics-file",
        str(metrics_file),
        "--poll-seconds",
        str(poll_seconds),
        "--cwd",
        str(repo_root),
    ]
    if env_json:
        full_cmd += ["--env-json", env_json]
    full_cmd += ["--"] + cmd
    print(f"\n[{stage}] gpu_set={gpu_set}")
    print(" ".join(full_cmd))
    rc = subprocess.call(full_cmd)
    metrics = {}
    if Path(metrics_file).exists():
        metrics = json.loads(Path(metrics_file).read_text())
    return rc, metrics


def find_latest_training_dir(output_root, model_lower, ratio, dataset, seq_len):
    pattern = f"{output_root}/training_output/{model_lower}/Diff-*-{ratio}_{dataset}_{seq_len}_*"
    cands = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return Path(cands[0]).name if cands else ""


def parse_ppl_from_log(log_path):
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Perplexity on .* is:\s*([0-9.]+)", text)
    return float(m.group(1)) if m else None


def canonical_model_name(model_lower):
    base = model_lower.split("__")[-1]
    return re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")


def ratio_to_str(ratio):
    return f"{ratio:g}"


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    monitor_script = repo_root / "scripts" / "run_with_gpu_monitor.py"
    if not monitor_script.exists():
        print(f"Missing {monitor_script}", file=sys.stderr)
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"matrix_{ts}"
    logs_root = repo_root / "logs" / run_name
    metrics_root = logs_root / "metrics"
    ensure_dir(logs_root)
    ensure_dir(metrics_root)

    summary_csv = logs_root / "summary.csv"
    summary_fields = [
        "model_id",
        "ratio",
        "trainer_status",
        "trainer_gpu_set",
        "trainer_peak_mem_mib_total",
        "trainer_oom",
        "training_result_path",
        "updater_status",
        "updater_gpu_set",
        "updater_peak_mem_mib_total",
        "updater_oom",
        "evaluate_status",
        "evaluate_gpu_set",
        "evaluate_peak_mem_mib_total",
        "ppl_wikitext2",
        "notes",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()

    ld_extra = conda_nvidia_lib_path()
    base_env_extra = {}
    if ld_extra:
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        base_env_extra["LD_LIBRARY_PATH"] = f"{ld_extra}:{prev}" if prev else ld_extra
    # Force unbuffered python logs so Slurm output reflects live progress.
    base_env_extra["PYTHONUNBUFFERED"] = "1"
    # Disable wandb auto-init so trainer runs are non-interactive and reproducible on cluster/local shells.
    base_env_extra["WANDB_DISABLED"] = "true"

    # flatten gpu ids from tries for updater fallback
    unique_gpus = []
    for gs in args.gpu_tries:
        for g in gs.split(","):
            g = g.strip()
            if g and g not in unique_gpus:
                unique_gpus.append(g)

    for model_id in args.models:
        model_lower = model_id.split("/")[-1]
        model_alias = canonical_model_name(model_lower)
        for ratio in args.ratios:
            row = {k: "" for k in summary_fields}
            row["model_id"] = model_id
            row["ratio"] = ratio
            notes = []

            trainer_ok = False
            trainer_metrics = {}
            trainer_gpu = ""
            for gpu_set in args.gpu_tries:
                stage = f"{model_lower}_r{ratio}_trainer"
                log_file = logs_root / f"{stage}_g{gpu_set.replace(',', '-')}.log"
                metrics_file = metrics_root / f"{stage}_g{gpu_set.replace(',', '-')}.json"
                trainer_cmd = [
                    args.python_bin,
                    "svd_trainer.py",
                    "--model_id",
                    model_id,
                    "--target_ratio",
                    str(ratio),
                    "--seq_len",
                    str(args.seq_len),
                    "--training_dataset",
                    args.training_dataset,
                    "--n_train_epochs",
                    str(args.n_train_epochs),
                    "--n_train_samples",
                    str(args.n_train_samples),
                    "--n_eval_samples",
                    str(args.n_eval_samples),
                    "--seed",
                    str(args.seed),
                    "--path_head_folder",
                    args.path_head_folder,
                    "--path_head_folder_output",
                    args.path_head_folder_output,
                ]
                if args.remapping:
                    trainer_cmd.append("--remapping")
                rc, m = run_monitored(
                    repo_root,
                    args.python_bin,
                    str(monitor_script),
                    "trainer",
                    gpu_set,
                    log_file,
                    metrics_file,
                    args.poll_seconds,
                    base_env_extra,
                    trainer_cmd,
                )
                if rc == 0:
                    trainer_ok = True
                    trainer_metrics = m
                    trainer_gpu = gpu_set
                    break
                if not m.get("oom_detected", False):
                    notes.append(f"trainer_non_oom_failure_gpu_{gpu_set}")
                    break
                notes.append(f"trainer_oom_gpu_{gpu_set}")

            row["trainer_status"] = "ok" if trainer_ok else "failed"
            row["trainer_gpu_set"] = trainer_gpu
            row["trainer_peak_mem_mib_total"] = trainer_metrics.get("max_memory_mib_total", "")
            row["trainer_oom"] = trainer_metrics.get("oom_detected", "")

            if not trainer_ok:
                row["notes"] = ";".join(notes)
                with open(summary_csv, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=summary_fields).writerow(row)
                continue

            train_result = find_latest_training_dir(
                args.path_head_folder_output, model_lower, ratio, args.training_dataset, args.seq_len
            )
            row["training_result_path"] = train_result
            if not train_result:
                row["updater_status"] = "failed"
                notes.append("cannot_find_training_result_path")
                row["notes"] = ";".join(notes)
                with open(summary_csv, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=summary_fields).writerow(row)
                continue

            updater_ok = False
            updater_metrics = {}
            updater_gpu = ""
            updater_try_order = [trainer_gpu.split(",")[0]] if trainer_gpu else []
            for g in unique_gpus:
                if g not in updater_try_order:
                    updater_try_order.append(g)

            for g in updater_try_order:
                stage = f"{model_lower}_r{ratio}_updater"
                log_file = logs_root / f"{stage}_g{g}.log"
                metrics_file = metrics_root / f"{stage}_g{g}.json"
                updater_cmd = [
                    args.python_bin,
                    "weight_updater.py",
                    "--model_id",
                    model_id,
                    "--training_result_path",
                    train_result,
                    "--seed",
                    str(args.seed),
                    "--n_train_samples",
                    str(args.n_train_samples),
                    "--n_eval_samples",
                    str(args.n_eval_samples),
                    "--training_dataset",
                    args.training_dataset,
                    "--path_head_folder",
                    args.path_head_folder,
                    "--path_head_folder_output",
                    args.path_head_folder_output,
                    "--owner",
                    args.owner,
                    "--model_alias",
                    model_alias,
                    "--method_name",
                    args.method_name,
                    "--output_format",
                    args.output_format,
                ]
                if args.remapping:
                    updater_cmd.append("--remapping")
                updater_env = dict(base_env_extra)
                if args.max_gamma > 0:
                    updater_env["DOBI_MAX_GAMMA"] = str(args.max_gamma)
                rc, m = run_monitored(
                    repo_root,
                    args.python_bin,
                    str(monitor_script),
                    "updater",
                    g,
                    log_file,
                    metrics_file,
                    args.poll_seconds,
                    updater_env,
                    updater_cmd,
                )
                if rc == 0:
                    updater_ok = True
                    updater_metrics = m
                    updater_gpu = g
                    break
                if not m.get("oom_detected", False):
                    notes.append(f"updater_non_oom_failure_gpu_{g}")
                    break
                notes.append(f"updater_oom_gpu_{g}")

            row["updater_status"] = "ok" if updater_ok else "failed"
            row["updater_gpu_set"] = updater_gpu
            row["updater_peak_mem_mib_total"] = updater_metrics.get("max_memory_mib_total", "")
            row["updater_oom"] = updater_metrics.get("oom_detected", "")

            if not updater_ok:
                row["notes"] = ";".join(notes)
                with open(summary_csv, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=summary_fields).writerow(row)
                continue

            if args.skip_eval:
                row["evaluate_status"] = "skipped"
                row["notes"] = ";".join(notes)
                with open(summary_csv, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=summary_fields).writerow(row)
                continue

            if args.remapping:
                model_out = Path(args.path_head_folder_output) / "compressed_model" / model_lower
                updated_model_path = model_out / f"DobiSVD-{model_lower}-{ratio}"
            else:
                ratio_str = ratio_to_str(ratio)
                model_out = Path(args.path_head_folder_output) / "compressed_model" / model_alias
                updated_model_path = (
                    model_out / args.method_name / f"{args.owner}_{model_alias}_{args.method_name}_{ratio_str}.{args.output_format}"
                )

            eval_gpu = updater_gpu if updater_gpu else (trainer_gpu.split(",")[0] if trainer_gpu else "0")
            stage = f"{model_lower}_r{ratio}_evaluate"
            log_file = logs_root / f"{stage}_g{eval_gpu}.log"
            metrics_file = metrics_root / f"{stage}_g{eval_gpu}.json"
            eval_cmd = [
                args.python_bin,
                "evaluate.py",
                "--updated_model_path",
                str(updated_model_path),
                "--eval_metric",
                "ppl",
                "--eval_dataset",
                "wikitext2",
                "--seq_len",
                str(args.seq_len),
                "--n_eval_samples",
                str(args.n_eval_samples),
                "--path_head_folder",
                args.path_head_folder,
                "--path_head_folder_output",
                args.path_head_folder_output,
            ]
            if args.remapping:
                eval_cmd.append("--remapping")
            rc, m = run_monitored(
                repo_root,
                args.python_bin,
                str(monitor_script),
                "evaluate",
                eval_gpu,
                log_file,
                metrics_file,
                args.poll_seconds,
                base_env_extra,
                eval_cmd,
            )
            row["evaluate_status"] = "ok" if rc == 0 else "failed"
            row["evaluate_gpu_set"] = eval_gpu
            row["evaluate_peak_mem_mib_total"] = m.get("max_memory_mib_total", "")
            if rc == 0 and Path(log_file).exists():
                ppl = parse_ppl_from_log(log_file)
                row["ppl_wikitext2"] = "" if ppl is None else ppl
            else:
                if m.get("oom_detected", False):
                    notes.append(f"evaluate_oom_gpu_{eval_gpu}")

            row["notes"] = ";".join(notes)
            with open(summary_csv, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=summary_fields).writerow(row)

    print(f"\nDone. Summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
