#!/usr/bin/env python3
# coding: utf-8

import sys
import argparse
import json
import os
import time
import resource
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(".")


from datautils import get_calib_data
from act_aware_utils import calib_input_distribution, calib_fisher_info
from sensitivity import calib_sensitivity_ppl, calib_sensitivity_stable_rank
from quantization import rtn_quant_sequential
from binary_search import binary_search_truncation_rank
from modules.svd_linear import SVDLinear


# ------------------------
# Timing + memory helpers (DF-SVD-style)
# ------------------------
def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


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


def _cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_sync_all() -> None:
    if not _cuda_available():
        return
    try:
        n = int(torch.cuda.device_count())
    except Exception:
        n = 1
    for i in range(max(1, n)):
        try:
            torch.cuda.synchronize(i)
        except Exception:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass


def _reset_cuda_peak_stats_all() -> None:
    if not _cuda_available():
        return
    try:
        n = int(torch.cuda.device_count())
    except Exception:
        n = 1
    for i in range(max(1, n)):
        try:
            torch.cuda.reset_peak_memory_stats(i)
        except Exception:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass


def _cuda_mem_snapshot_all() -> Dict[str, Dict[str, int]]:
    """Snapshot current+peak CUDA stats for every visible device."""
    if not _cuda_available():
        return {}
    snap: Dict[str, Dict[str, int]] = {}
    try:
        n = int(torch.cuda.device_count())
    except Exception:
        n = 1
    for i in range(max(1, n)):
        try:
            free, total = torch.cuda.mem_get_info(i)
        except Exception:
            try:
                free, total = torch.cuda.mem_get_info()
            except Exception:
                free, total = 0, 0
        try:
            alloc = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            max_alloc = torch.cuda.max_memory_allocated(i)
            max_reserved = torch.cuda.max_memory_reserved(i)
        except Exception:
            try:
                alloc = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                max_alloc = torch.cuda.max_memory_allocated()
                max_reserved = torch.cuda.max_memory_reserved()
            except Exception:
                alloc = reserved = max_alloc = max_reserved = 0
        snap[str(i)] = {
            "alloc_bytes": int(alloc),
            "reserved_bytes": int(reserved),
            "max_alloc_bytes": int(max_alloc),
            "max_reserved_bytes": int(max_reserved),
            "free_bytes": int(free),
            "total_bytes": int(total),
        }
    return snap


def _model_size_bytes(model: torch.nn.Module) -> Dict[str, int]:
    """Persistent model footprint from parameters + buffers."""
    param_bytes = 0
    param_count = 0
    for p in model.parameters():
        try:
            param_count += int(p.numel())
            param_bytes += int(p.numel()) * int(p.element_size())
        except Exception:
            continue
    buffer_bytes = 0
    buffer_count = 0
    for b in model.buffers():
        try:
            buffer_count += int(b.numel())
            buffer_bytes += int(b.numel()) * int(b.element_size())
        except Exception:
            continue
    return {
        "param_bytes": int(param_bytes),
        "buffer_bytes": int(buffer_bytes),
        "param_count": int(param_count),
        "buffer_count": int(buffer_count),
    }


def _ensure_tokenizer(tok: Any, model_id: str, *, hf_token: Optional[str] = None) -> Any:
    """Return a callable HF tokenizer; reload if tok is invalid (e.g., bool)."""
    try:
        if tok is not None and not isinstance(tok, bool) and callable(tok):
            return tok
    except Exception:
        pass

    kwargs: Dict[str, Any] = {"trust_remote_code": True, "use_fast": False}
    if hf_token:
        # Transformers has used both 'token' and 'use_auth_token' across versions.
        kwargs["token"] = hf_token

    try:
        tok2 = AutoTokenizer.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop("token", None)
        if hf_token:
            kwargs["use_auth_token"] = hf_token
        tok2 = AutoTokenizer.from_pretrained(model_id, **kwargs)

    # LLaMA-family often has no pad token by default.
    try:
        if getattr(tok2, "pad_token", None) is None and getattr(tok2, "eos_token", None) is not None:
            tok2.pad_token = tok2.eos_token
    except Exception:
        pass
    return tok2


def _timing_write(out_dir: str, filename: str, timing: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)
    return path


def _run_stage(timing: Dict[str, Any], name: str, fn: Callable[[], Any]) -> Any:
    """Run `fn` and record synchronized wall-clock time + CUDA peak memory."""
    _cuda_sync_all()
    _reset_cuda_peak_stats_all()
    mem_before = _cuda_mem_snapshot_all()
    t0 = time.perf_counter()
    out = fn()
    _cuda_sync_all()
    sec = time.perf_counter() - t0
    mem_after = _cuda_mem_snapshot_all()
    timing["stages"].append(
        {
            "name": str(name),
            "sec": float(sec),
            "gpu_mem_before": mem_before,
            "gpu_mem_after": mem_after,
        }
    )
    return out


def _call_get_calib_data(
    dataset: str,
    tokenizer: Any,
    model_id: str,
    nsamples: int,
    *,
    seqlen: Optional[int],
    seed: Optional[int],
    use_bos: Optional[bool],
):
    """Call get_calib_data with best-effort signature compatibility.

    IMPORTANT: Only retry on *signature mismatch* TypeErrors.
    If get_calib_data throws a runtime TypeError (e.g. tokenizer becomes bool),
    we re-raise so the real bug isn't masked and retried 5 times.
    """

    def _sig_mismatch(err: TypeError) -> bool:
        msg = str(err)
        return (
            "unexpected keyword argument" in msg
            or ("positional" in msg and "argument" in msg)
            or ("takes" in msg and "positional" in msg)
        )

    # Ensure tokenizer is callable before calling into datautils.
    try:
        tokenizer = _ensure_tokenizer(tokenizer, model_id)
    except Exception:
        # Fall through; datautils may still handle it, but we won't mask non-signature errors.
        pass

    # Try keyword-rich call first (most explicit / least error-prone).
    # first lines of get_calib_data()
    print("calib entry:", __file__, type(tokenizer), callable(tokenizer), repr(tokenizer))
    assert not isinstance(tokenizer, bool), f"bad tokenizer at entry: {tokenizer!r}"
    try:
        return get_calib_data(
            dataset,
            tokenizer,
            model_id,
            nsamples,
            seqlen=seqlen,
            seed=seed,
            use_bos=use_bos,
        )
    except TypeError as e:
        if not _sig_mismatch(e):
            raise

    # Try without use_bos
    try:
        return get_calib_data(
            dataset,
            tokenizer,
            model_id,
            nsamples,
            seqlen=seqlen,
            seed=seed,
        )
    except TypeError as e:
        if not _sig_mismatch(e):
            raise

    # Try positional fallbacks
    if seqlen is not None and seed is not None and use_bos is not None:
        try:
            return get_calib_data(dataset, tokenizer, model_id, nsamples, int(seqlen), int(seed), bool(use_bos))
        except TypeError as e:
            if not _sig_mismatch(e):
                raise

    if seqlen is not None and seed is not None:
        try:
            return get_calib_data(dataset, tokenizer, model_id, nsamples, int(seqlen), int(seed))
        except TypeError as e:
            if not _sig_mismatch(e):
                raise

    # Final minimal call
    return get_calib_data(dataset, tokenizer, model_id, nsamples)


def main(args):
    run_start = time.perf_counter()
    timing: Dict[str, Any] = {
        "started_at": _now_iso(),
        "args": vars(args),
        "stages": [],
    }

    # Validate key args (keep build_asvd_repo flexible; avoid hard asserts).
    if args.param_ratio_target <= 0:
        raise ValueError("--param_ratio_target must be > 0 (e.g., 0.4 for 40% params kept).")

    # Compute where the HF repo will be written (also used as default timing_dir).
    save_path = (
        args.save_path
        if args.save_path
        else os.path.join(
            "huggingface_repos",
            args.model_id.split("/")[-1] + f"-asvd{int(args.param_ratio_target * 100)}",
        )
    )
    timing["save_path"] = save_path

    # Record baseline memory.
    timing["gpu_mem_begin"] = _cuda_mem_snapshot_all()

    model_id = args.model_id

    try:
        # Load tokenizer
        def _load_tok():
            tok = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=False,
                token=args.hf_token,
            )
            return _ensure_tokenizer(tok, model_id, hf_token=args.hf_token)

        try:
            tokenizer = _run_stage(timing, "load_tokenizer", _load_tok)
        except TypeError:
            # Older transformers uses use_auth_token
            def _load_tok_compat():
                tok = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    use_fast=False,
                    use_auth_token=args.hf_token,
                )
                return _ensure_tokenizer(tok, model_id, hf_token=args.hf_token)

            tokenizer = _run_stage(timing, "load_tokenizer", _load_tok_compat)
        # after tokenizer load in build_asvd_repo.py
        print("build tokenizer:", __file__, type(tokenizer), callable(tokenizer), repr(tokenizer))
        assert not isinstance(tokenizer, bool), f"bad tokenizer in build: {tokenizer!r}"
        # Load model
        def _load_model():
            try:
                return AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    token=args.hf_token,
                )
            except TypeError:
                # Fallbacks for older transformers keyword args
                try:
                    return AutoModelForCausalLM.from_pretrained(
                        model_id,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        use_auth_token=args.hf_token,
                    )
                except TypeError:
                    return AutoModelForCausalLM.from_pretrained(
                        model_id,
                        device_map="auto",
                        dtype=torch.float16,  # very old versions
                        trust_remote_code=True,
                    )

        model = _run_stage(timing, "load_model", _load_model)

        # Persistent model footprint (params + buffers).
        try:
            ms = _model_size_bytes(model)
            timing["model_param_bytes"] = int(ms.get("param_bytes", 0))
            timing["model_buffer_bytes"] = int(ms.get("buffer_bytes", 0))
            timing["model_param_count"] = int(ms.get("param_count", 0))
            timing["model_buffer_count"] = int(ms.get("buffer_count", 0))
        except Exception:
            pass

        # Calibration loader
        tokenizer = _ensure_tokenizer(tokenizer, model_id, hf_token=args.hf_token)

        def _build_calib():
            return _call_get_calib_data(
                args.calib_dataset,
                tokenizer,
                model_id,
                int(args.n_calib_samples),
                seqlen=args.calib_seqlen,
                seed=args.seed,
                use_bos=args.use_bos,
            )

        calib_loader = _run_stage(timing, "get_calib_data", _build_calib)

        # Scaling stats
        if "fisher" in args.scaling_method:
            _run_stage(timing, "calib_fisher_info", lambda: calib_fisher_info(model, calib_loader, args.use_cache))
        if "abs" in args.scaling_method:
            _run_stage(
                timing,
                f"calib_input_distribution_{args.scaling_method}",
                lambda: calib_input_distribution(model, calib_loader, args.scaling_method, args.use_cache),
            )

        # Sensitivity
        if args.sensitivity_metric == "ppl":
            sensitivity = _run_stage(
                timing, "calib_sensitivity_ppl", lambda: calib_sensitivity_ppl(model, calib_loader, args, args.use_cache)
            )
        elif args.sensitivity_metric == "stable_rank":
            sensitivity = _run_stage(
                timing,
                "calib_sensitivity_stable_rank",
                lambda: calib_sensitivity_stable_rank(model, calib_loader, args, args.use_cache),
            )
        else:
            raise ValueError(f"Unknown sensitivity_metric: {args.sensitivity_metric}")

        # Rank search
        _run_stage(
            timing,
            "binary_search_truncation_rank",
            lambda: binary_search_truncation_rank(model, sensitivity, calib_loader, args),
        )

        # Optional quantization (kept for compatibility with existing args)
        if args.weight_quant != "none":
            if args.weight_quant == "rtn_int8":
                _run_stage(timing, "rtn_quant_int8", lambda: rtn_quant_sequential(model, 8))
            elif args.weight_quant == "rtn_int6":
                _run_stage(timing, "rtn_quant_int6", lambda: rtn_quant_sequential(model, 6))
            else:
                raise ValueError(f"Unsupported weight_quant={args.weight_quant} in build_asvd_repo.py")

        # Save HF repo
        os.makedirs(save_path, exist_ok=True)

        _run_stage(timing, "save_tokenizer", lambda: tokenizer.save_pretrained(save_path))
        _run_stage(timing, "save_model", lambda: model.save_pretrained(save_path))

        # Patch config with truncation ranks
        def _write_config():
            config = model.config.to_dict()
            config["truncation_ranks"] = {}
            for name, module in model.named_modules():
                if isinstance(module, SVDLinear):
                    config["truncation_ranks"][name] = int(module.truncation_rank)
            # auto_map injection for custom modeling files
            if "opt" in model_id:
                config["auto_map"] = {
                    "AutoConfig": "configuration_asvd_opt.ASVDOPTConfig",
                    "AutoModelForCausalLM": "modeling_asvd_opt.ASVDOPTForCausalLM",
                }
                config["architectures"] = ["ASVDOPTForCausalLM"]
                os.system(
                    "cp ./huggingface_repos/configuration_asvd_opt.py ./huggingface_repos/modeling_asvd_opt.py ./"
                    + save_path
                )
            elif "llama" in model_id:
                config["auto_map"] = {
                    "AutoConfig": "configuration_asvd_llama.ASVDLlamaConfig",
                    "AutoModelForCausalLM": "modeling_asvd_llama.ASVDLlamaForCausalLM",
                }
                config["architectures"] = ["ASVDLlamaForCausalLM"]
                os.system(
                    "cp ./huggingface_repos/configuration_asvd_llama.py ./huggingface_repos/modeling_asvd_llama.py ./"
                    + save_path
                )

            with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

        _run_stage(timing, "write_config_json", _write_config)

        print(f"Done building huggingface model at: {save_path}", flush=True)

        # Optional push to hub (kept same semantics)
        if args.push:
            def _push():
                hub_name = model_id.split("/")[-1] + f"-asvd{int(args.param_ratio_target * 100)}"
                tok2 = AutoTokenizer.from_pretrained(save_path, trust_remote_code=True)
                tok2 = _ensure_tokenizer(tok2, save_path, hf_token=args.hf_token)
                mdl2 = AutoModelForCausalLM.from_pretrained(
                    save_path,
                    device_map="cpu",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                tok2.push_to_hub(hub_name)
                mdl2.push_to_hub(hub_name)

            _run_stage(timing, "push_to_hub", _push)

    finally:
        _cuda_sync_all()
        timing["ended_at"] = _now_iso()
        timing["total_sec"] = float(time.perf_counter() - run_start)

        # Peak CPU RSS
        rss = _cpu_maxrss_bytes()
        if rss is not None:
            timing["cpu_maxrss_bytes"] = int(rss)

        # Final GPU mem snapshot + overall peaks across stages
        timing["gpu_mem_end"] = _cuda_mem_snapshot_all()

        # Compute global GPU peak (max over stages + devices)
        try:
            peak_alloc = 0
            peak_reserved = 0
            for st in timing.get("stages", []):
                m = st.get("gpu_mem_after", {})
                if isinstance(m, dict):
                    for _dev, dd in m.items():
                        if isinstance(dd, dict):
                            peak_alloc = max(peak_alloc, int(dd.get("max_alloc_bytes", 0)))
                            peak_reserved = max(peak_reserved, int(dd.get("max_reserved_bytes", 0)))
            timing["gpu_peak_alloc_bytes"] = int(peak_alloc)
            timing["gpu_peak_reserved_bytes"] = int(peak_reserved)
        except Exception:
            pass

        # Write timing JSON (default into save_path).
        out_dir = args.timing_dir if args.timing_dir else save_path
        try:
            timing_path = _timing_write(out_dir, args.timing_file, timing)
            print(f"[Timing] wrote: {timing_path}", flush=True)
            if isinstance(timing.get("gpu_peak_alloc_bytes"), int) and isinstance(timing.get("gpu_peak_reserved_bytes"), int):
                ga = float(timing["gpu_peak_alloc_bytes"]) / (1024.0 ** 3)
                gr = float(timing["gpu_peak_reserved_bytes"]) / (1024.0 ** 3)
                print(f"[Mem] gpu_peak_alloc={ga:.2f}GB gpu_peak_reserved={gr:.2f}GB", flush=True)
            print(f"[Time] total={timing.get('total_sec', 0.0):.2f}s", flush=True)
        except Exception as e:
            print(f"[Warn] Failed to write timing JSON: {e}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="facebook/opt-1.3b", help="Pretrained model ID")
    parser.add_argument("--ppl_target", type=float, default=-1, help="target ppl")
    parser.add_argument("--param_ratio_target", type=float, default=-1, help="target param ratio")
    parser.add_argument("--act_aware", action="store_true", help="use act aware svd (ASVD)")
    parser.add_argument("--alpha", type=float, default=0.5, help="hyper-parameter alpha for ASVD")

    # calibration
    parser.add_argument("--n_calib_samples", type=int, default=256, help="number of samples used for calibration")
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb"],
        help="calibration dataset",
    )
    parser.add_argument("--calib_seqlen", type=int, default=2048, help="calibration sequence length (best-effort)")
    parser.add_argument("--seed", type=int, default=3, help="calibration sampling seed (best-effort)")
    parser.add_argument("--use_bos", action="store_true", help="use BOS token in calibration if supported")

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
    parser.add_argument("--use_cache", action="store_true", help="use cached calibration results")
    parser.add_argument(
        "--weight_quant",
        type=str,
        default="none",
        choices=["none", "rtn_int8", "rtn_int6"],
        help="weight quantization method",
    )

    parser.add_argument(
        "--eval_mmlu",
        action="store_true",
        help="(compat) evaluate mmlu (unused in build_asvd_repo)",
    )

    # args referenced inside sensitivity/binary_search
    parser.add_argument("--compress_kv_cache", action="store_true", help="compress kv cache by asvd for k_proj and v_proj")
    parser.add_argument("--kv_cache_ratio_target", type=float, default=-1, help="kv cache ratio")
    parser.add_argument("--rank_align", type=int, default=1, help="align rank in SVD")
    parser.add_argument("--sigma_fuse", type=str, default="UV", choices=["U", "V", "UV"], help="sigma fuse method")

    # output / hub
    parser.add_argument("--save_path", type=str, default=None, help="Override HF repo save path (default: huggingface_repos/<name>-asvdXX)")
    parser.add_argument("--push", action="store_true", help="push to hub")
    parser.add_argument("--hf_token", type=str, default=None, help="HF token (optional, for gated/private repos)")

    # timing
    parser.add_argument("--timing_dir", type=str, default=None, help="Where to write timing JSON (default: save_path)")
    parser.add_argument("--timing_file", type=str, default="asvd_build_timing.json", help="Timing JSON filename")

    args = parser.parse_args()
    main(args)
