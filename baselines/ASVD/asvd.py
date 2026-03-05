#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASVD / Act-Aware SVD baseline script with DF-SVD-style timing + memory instrumentation.

This keeps the original behavior (calibration -> rank search -> optional quantization -> evaluation),
but records:
  - wall-clock stage timings with CUDA synchronization (more accurate GPU timings)
  - CUDA memory snapshots (allocated / reserved + peak allocated / peak reserved) per visible GPU
  - best-effort peak CPU RSS (ru_maxrss)

Outputs:
  - result text: <output_dir>/result.txt
  - timing json: <output_dir>/<timing_file>
"""

import argparse
import json
import os
import resource
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_utils import evaluate_model
from datautils import get_calib_data
from act_aware_utils import calib_input_distribution, calib_fisher_info
from sensitivity import calib_sensitivity_ppl, calib_sensitivity_stable_rank
from quantization import rtn_quant_sequential, awq_quant_sequential
from binary_search import binary_search_truncation_rank


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _timing_write(out_dir: str, filename: str, timing: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)
    return path


def _cpu_maxrss_bytes() -> Optional[int]:
    """Best-effort peak CPU RSS in bytes."""
    try:
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss: KB on Linux, bytes on macOS.
        if sys.platform == "darwin":
            return int(r)
        return int(r * 1024)
    except Exception:
        return None


def _cuda_devices() -> List[torch.device]:
    if not torch.cuda.is_available():
        return []
    return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]


def _cuda_sync_all() -> None:
    if not torch.cuda.is_available():
        return
    for d in _cuda_devices():
        try:
            torch.cuda.synchronize(d)
        except Exception:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass


def _reset_cuda_peak_stats_all() -> None:
    if not torch.cuda.is_available():
        return
    for d in _cuda_devices():
        try:
            torch.cuda.reset_peak_memory_stats(d)
        except Exception:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass


def _cuda_mem_snapshot(device: torch.device) -> Dict[str, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    try:
        free, total = torch.cuda.mem_get_info(device)
    except Exception:
        free, total = torch.cuda.mem_get_info()
    try:
        alloc = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_alloc = torch.cuda.max_memory_allocated(device)
        max_reserved = torch.cuda.max_memory_reserved(device)
    except Exception:
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_alloc = torch.cuda.max_memory_allocated()
        max_reserved = torch.cuda.max_memory_reserved()
    return {
        "alloc_bytes": int(alloc),
        "reserved_bytes": int(reserved),
        "max_alloc_bytes": int(max_alloc),
        "max_reserved_bytes": int(max_reserved),
        "free_bytes": int(free),
        "total_bytes": int(total),
    }


def _cuda_mem_snapshot_all() -> Dict[str, Dict[str, int]]:
    snap: Dict[str, Dict[str, int]] = {}
    for d in _cuda_devices():
        snap[str(d)] = _cuda_mem_snapshot(d)
    return snap


def _record_stage(timing: Dict[str, Any], name: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and record synchronized timing + CUDA memory."""
    _cuda_sync_all()
    _reset_cuda_peak_stats_all()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    _cuda_sync_all()
    sec = time.perf_counter() - t0
    timing["stages"].append(
        {
            "name": str(name),
            "sec": float(sec),
            "gpu_mem": _cuda_mem_snapshot_all(),
            "cpu_maxrss_bytes": _cpu_maxrss_bytes(),
        }
    )
    return out


def main(args):
    # setting random seed of numpy and torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    run_start = time.perf_counter()
    timing: Dict[str, Any] = {
        "started_at": _now_iso(),
        "args": vars(args),
        "stages": [],
        "result": None,
        "paths": {},
    }
    timing["gpu_mem_begin"] = _cuda_mem_snapshot_all()
    timing["cpu_maxrss_bytes_begin"] = _cpu_maxrss_bytes()

    # Load tokenizer
    model_id = args.model_id
    tokenizer = _record_stage(
        timing,
        "load_tokenizer",
        AutoTokenizer.from_pretrained,
        model_id,
        trust_remote_code=True,
    )

    # Load model
    model = _record_stage(
        timing,
        "load_model",
        AutoModelForCausalLM.from_pretrained,
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Calibration / compression
    if not args.raw_model:
        calib_loader = _record_stage(
            timing,
            "build_calib_loader",
            get_calib_data,
            args.calib_dataset,
            tokenizer,
            model_id,
            args.n_calib_samples,
            seed=args.seed,
            use_bos=args.use_bos,
        )

        if "fisher" in args.scaling_method:
            _record_stage(timing, "calib_fisher_info", calib_fisher_info, model, calib_loader, args.use_cache)

        if "abs" in args.scaling_method:
            _record_stage(
                timing,
                "calib_input_distribution",
                calib_input_distribution,
                model,
                calib_loader,
                args.scaling_method,
                args.use_cache,
            )

        if args.sensitivity_metric == "ppl":
            sensitivity = _record_stage(
                timing,
                "calib_sensitivity_ppl",
                calib_sensitivity_ppl,
                model,
                calib_loader,
                args,
                args.use_cache,
            )
        elif args.sensitivity_metric == "stable_rank":
            sensitivity = _record_stage(
                timing,
                "calib_sensitivity_stable_rank",
                calib_sensitivity_stable_rank,
                model,
                calib_loader,
                args,
                args.use_cache,
            )
        else:
            raise ValueError(f"Unknown sensitivity_metric: {args.sensitivity_metric}")

        _record_stage(
            timing,
            "binary_search_truncation_rank",
            binary_search_truncation_rank,
            model,
            sensitivity,
            calib_loader,
            args,
        )

        if args.weight_quant != "none":
            if args.weight_quant == "rtn_int8":
                _record_stage(timing, "quant_rtn_int8", rtn_quant_sequential, model, 8)
            elif args.weight_quant == "rtn_int6":
                _record_stage(timing, "quant_rtn_int6", rtn_quant_sequential, model, 6)
            elif args.weight_quant == "awq_int8":
                model = _record_stage(timing, "quant_awq_int8", awq_quant_sequential, model, tokenizer, 8)
            elif args.weight_quant == "awq_int4":
                model = _record_stage(timing, "quant_awq_int4", awq_quant_sequential, model, tokenizer, 4)

    # Evaluate
    result = _record_stage(
        timing,
        "evaluate_model",
        evaluate_model,
        model,
        tokenizer,
        args.model_id,
        "mmlu" if args.eval_mmlu else args.eval_tasks,
        eval_ppl=args.eval_ppl,
        limit=-1,
        use_bos=args.use_bos,
    )
    timing["result"] = result
    print(result)

    # Persist results
    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, "result.txt")
    with open(result_path, "a+", encoding="utf-8") as f:
        f.write(f"{args}\n")
        f.write(f"{result}\n")
    timing["paths"]["result_txt"] = result_path

    # Finalize timing
    timing["ended_at"] = _now_iso()
    timing["total_sec"] = float(time.perf_counter() - run_start)
    timing["gpu_mem_end"] = _cuda_mem_snapshot_all()
    timing["cpu_maxrss_bytes_end"] = _cpu_maxrss_bytes()

    # Aggregate peak GPU usage across stages
    peaks_by_dev: Dict[str, Dict[str, int]] = {}
    for st in timing.get("stages", []):
        gm = st.get("gpu_mem", {}) if isinstance(st, dict) else {}
        if not isinstance(gm, dict):
            continue
        for dev, mem in gm.items():
            if not isinstance(mem, dict):
                continue
            ma = int(mem.get("max_alloc_bytes", 0))
            mr = int(mem.get("max_reserved_bytes", 0))
            cur = peaks_by_dev.get(dev, {"max_alloc_bytes": 0, "max_reserved_bytes": 0})
            if ma > int(cur.get("max_alloc_bytes", 0)):
                cur["max_alloc_bytes"] = ma
            if mr > int(cur.get("max_reserved_bytes", 0)):
                cur["max_reserved_bytes"] = mr
            peaks_by_dev[dev] = cur
    timing["gpu_peak_by_device"] = peaks_by_dev
    timing["gpu_peak_alloc_bytes"] = int(max([v.get("max_alloc_bytes", 0) for v in peaks_by_dev.values()] + [0]))
    timing["gpu_peak_reserved_bytes"] = int(
        max([v.get("max_reserved_bytes", 0) for v in peaks_by_dev.values()] + [0])
    )

    timing_path = _timing_write(args.output_dir, args.timing_file, timing)
    timing["paths"]["timing_json"] = timing_path
    print(f"[Time] total={timing['total_sec']:.2f}s timing_json={timing_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--ppl_target",
        type=float,
        default=-1,
        help="target ppl",
    )
    parser.add_argument(
        "--param_ratio_target",
        type=float,
        default=-1,
        help="target param ratio",
    )
    parser.add_argument(
        "--act_aware",
        action="store_true",
        help="use act aware svd (ASVD)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="hyper-parameter alpha for ASVD",
    )
    parser.add_argument(
        "--n_calib_samples",
        type=int,
        default=32,
        help="number of samples used for calibration",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb", "alpaca", "selfgen"],
        help="calibration dataset",
    )
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="abs_mean",
        choices=["abs_mean", "abs_max", "fisher", "fisher_abs_mean"],
        help="scaling method",
    )
    parser.add_argument(
        "--sensitivity_metric",
        type=str,
        default="ppl",
        choices=["ppl", "stable_rank"],
        help="search metric",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="use cached calibration results",
    )
    parser.add_argument(
        "--weight_quant",
        type=str,
        default="none",
        choices=["none", "rtn_int8", "rtn_int6", "awq_int8", "awq_int4"],
        help="weight quantization method",
    )
    parser.add_argument(
        "--eval_mmlu",
        action="store_true",
        help="evaluate mmlu",
    )
    parser.add_argument(
        "--eval_ppl",
        default="wikitext2,ptb",
        type=str,
    )
    parser.add_argument("--eval_tasks", type=str, default="")
    parser.add_argument(
        "--sigma_fuse",
        type=str,
        default="UV",
        help="sigma fuse method",
        choices=["U", "V", "UV"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=233,
        help="random seed, which can significantly affect the calibration results",
    )
    parser.add_argument(
        "--compress_kv_cache",
        action="store_true",
        help="compress kv cache by asvd for k_proj and v_proj",
    )
    parser.add_argument(
        "--kv_cache_ratio_target",
        type=float,
        default=-1,
        help="kv cache ratio",
    )
    parser.add_argument(
        "--rank_align",
        type=int,
        default=1,
        help="align rank in SVD",
    )
    parser.add_argument(
        "--raw_model",
        action="store_true",
        help="use the raw model without ASVD",
    )
    parser.add_argument(
        "--use_bos",
        action="store_true",
        help="use bos token in calibration",
    )

    # Instrumentation outputs
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to write result.txt + timing json.",
    )
    parser.add_argument(
        "--timing_file",
        type=str,
        default="asvd_timing.json",
        help="Filename for timing json written under --output_dir.",
    )

    args = parser.parse_args()
    main(args)
