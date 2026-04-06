"""
eval_decoder.py — unified decoder LLM evaluation.

Input:  one HF checkpoint directory (or HF model id for baseline)
Output: one appended row in a CSV  +  one JSON (lm-eval raw output)

Usage:
    python eval_decoder.py \
        --checkpoint /path/to/hf_dir \
        --model_family llama \
        --model_tag    Llama-3.1-8B \
        --is_instruct  0 \
        --method       SVDLLMv2 \
        --keep_ratio   0.8 \
        --dtype bf16 --device cuda:0 \
        --output_csv results/llama31_8b.csv

    # Baseline from HF Hub:
    python eval_decoder.py \
        --checkpoint   meta-llama/Llama-3.1-8B \
        --model_family llama \
        --model_tag    Llama-3.1-8B \
        --is_instruct  0 \
        --method baseline --keep_ratio 1.0 \
        --output_csv results/llama31_8b.csv

    # Skip one stage:
    python eval_decoder.py ... --no_ppl
    python eval_decoder.py ... --no_lmeval
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

# ── task configuration ────────────────────────────────────────────────────────

# Ordered task list
ORDERED_TASKS = [
    "boolq", "arc_challenge", "arc_easy", "hellaswag",
    "winogrande", "openbookqa", "piqa", "mathqa",
]

# Primary metric used for avg_score computation
PRIMARY_METRIC: dict[str, str] = {
    "boolq":         "acc",
    "arc_challenge": "acc_norm",
    "arc_easy":      "acc_norm",
    "hellaswag":     "acc_norm",
    "winogrande":    "acc",
    "openbookqa":    "acc_norm",
    "piqa":          "acc_norm",
    "mathqa":        "acc",
}

DEFAULT_TASKS    = ",".join(ORDERED_TASKS)
DEFAULT_DATASETS = "wikitext2,c4_stream,ptb"

METRIC_PROTOCOL = ";".join(f"{t}={PRIMARY_METRIC[t]}" for t in ORDERED_TASKS)

# Per-task CSV columns: {task}_acc, {task}_acc_norm, {task}_std
_TASK_COLS: list[str] = []
for _t in ORDERED_TASKS:
    _TASK_COLS += [f"{_t}_acc", f"{_t}_acc_norm", f"{_t}_std"]

CSV_FIELDS = [
    # identity
    "model_family", "model_tag", "is_instruct", "method", "keep_ratio", "dtype",
    # status
    "compression_success", "eval_success",
    # protocol
    "task_set", "metric_protocol",
    # PPL
    "wikitext2_ppl", "c4_ppl", "ptb_ppl",
    # zero-shot tasks (3 cols per task: acc, acc_norm, std)
    *_TASK_COLS,
    "avg_score",
    # checkpoint info
    "checkpoint_path", "checkpoint_size_gb",
    # run config
    "seq_len", "eval_batch_size", "device",
    # versions
    "lm_eval_version", "transformers_version",
    # misc
    "timestamp", "notes",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return "N/A" if math.isnan(v) else f"{v:.6f}"
    return str(v)


def _get_metric(task_result: dict, metric: str) -> float:
    """Handle lm_eval v0.3 ('acc') and v0.4+ ('acc,none') key formats."""
    v = task_result.get(f"{metric},none", task_result.get(metric))
    return float(v) if v is not None else float("nan")


def _get_stderr(task_result: dict, metric: str) -> float:
    """Return stderr for a metric; handles both v0.3 and v0.4+ key formats."""
    v = task_result.get(f"{metric},stderr",
        task_result.get(f"{metric}_stderr"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _dir_size_gb(path: str) -> float:
    """Total size of all files in a directory, in GB. Returns nan if not a dir."""
    p = Path(path)
    if not p.is_dir():
        return float("nan")
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return total / (1024 ** 3)


def _already_done(csv_path: str, model_tag: str, method: str,
                  keep_ratio: float) -> bool:
    """Return True if a row with matching (model_tag, method, keep_ratio) exists."""
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return False
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("model_tag") == model_tag
                    and row.get("method") == method
                    and float(row.get("keep_ratio", -1)) == keep_ratio):
                return True
    return False


def _lm_eval_version() -> str:
    try:
        import lm_eval
        return getattr(lm_eval, "__version__", "unknown")
    except ImportError:
        return "not_installed"


def _transformers_version() -> str:
    try:
        import transformers
        return transformers.__version__
    except ImportError:
        return "not_installed"


# ── loading ───────────────────────────────────────────────────────────────────

def _load_model_pt(pt_path: Path, dtype: torch.dtype, device: str):
    """Load a model from a model.pt file inside a checkpoint directory.

    Reads lowrank_config.json (if present) to determine which framework's
    classes need to be on sys.path before torch.load.  Falls back to the
    Dobi-SVD path for directories that have no lowrank_config.json.
    """
    import sys as _sys
    import json as _json
    import torch.nn as _nn
    import torch.nn.functional as _F

    # Shim 1: SiLUActivation removed in newer transformers — replace with a
    # compat class that has 'inplace' as a class-level attribute so it survives
    # unpickling with an empty __dict__.
    try:
        import transformers.activations as _act
        if not hasattr(_act, "SiLUActivation"):
            class _SiLUActivationCompat(_nn.Module):
                inplace = False
                def forward(self, x):
                    return _F.silu(x)
            _act.SiLUActivation = _SiLUActivationCompat
    except Exception:
        pass

    # Shim 2: ASVD SVDLinear stores plain nn.SiLU objects; older checkpoints
    # may lose the 'inplace' instance attr after unpickling.  Patch forward()
    # to use a getattr fallback so it never raises AttributeError.
    try:
        def _silu_forward_safe(self, x):
            return _F.silu(x, inplace=getattr(self, "inplace", False))
        _nn.SiLU.forward = _silu_forward_safe
    except Exception:
        pass

    root = Path(__file__).resolve().parent.parent
    ckpt_dir = pt_path.parent

    # Determine framework from lowrank_config.json
    cfg_file = ckpt_dir / "lowrank_config.json"
    framework = "dobi"  # default
    if cfg_file.exists():
        try:
            meta = _json.loads(cfg_file.read_text())
            framework = meta.get("framework", "dobi")
        except Exception:
            pass

    if framework == "svdllm":
        svdllm_dir = root / "baselines" / "SVD-LLM"
        for p in (str(svdllm_dir), str(svdllm_dir / "flashsvd_component")):
            if p not in _sys.path:
                _sys.path.insert(0, p)

    elif framework == "asvd":
        asvd_dir = root / "baselines" / "ASVD"
        if str(asvd_dir) not in _sys.path:
            _sys.path.insert(0, str(asvd_dir))

    else:  # dobi (or unknown)
        dobi_root = root / "baselines" / "Dobi-SVD"
        for p in (str(dobi_root), str(dobi_root / "modules")):
            if p not in _sys.path:
                _sys.path.insert(0, p)
        try:
            import modules.module as _  # noqa: F401
        except ImportError:
            try:
                import module as _  # noqa: F401
            except ImportError:
                pass

    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    model = obj["model"] if isinstance(obj, dict) else obj

    # Fix any nn.SiLU instances that lost 'inplace' after unpickling.
    for m in model.modules():
        if isinstance(m, _nn.SiLU) and not hasattr(m, "inplace"):
            m.inplace = False

    return model.to(dtype=dtype).to(device)


def _resolve_checkpoint(checkpoint: str) -> str:
    """If the checkpoint dir lacks config.json, descend to find it (handles
    nested save paths like .../outer/inner/inner where inner has the files)."""
    p = Path(checkpoint)
    if (p / "config.json").exists() or (p / "model.pt").exists():
        return checkpoint
    # BFS one level deep
    for child in sorted(p.rglob("config.json")):
        return str(child.parent)
    return checkpoint


def load_model(checkpoint: str, dtype: torch.dtype, device: str,
               hf_token: str | None, tokenizer_path: str | None = None):
    checkpoint = _resolve_checkpoint(checkpoint)
    ckpt_dir = Path(checkpoint)
    model_pt = ckpt_dir / "model.pt"

    # Directory has model.pt instead of standard safetensors/bin weights
    if model_pt.is_file():
        from transformers import AutoTokenizer
        extra = {"token": hf_token} if hf_token else {}
        tok_src = tokenizer_path or checkpoint
        tokenizer = AutoTokenizer.from_pretrained(
            tok_src, trust_remote_code=True, **extra
        )
        model = _load_model_pt(model_pt, dtype, device)
        return model, tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer
    extra = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True, **extra
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **extra,
    )
    return model, tokenizer


# ── PPL ───────────────────────────────────────────────────────────────────────

def _iter_texts(dataset_name: str):
    from datasets import load_dataset
    name = dataset_name.lower()
    if name in {"wikitext2", "wikitext-2", "wiki2"}:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        for ex in ds:
            if ex.get("text", "").strip():
                yield ex["text"]
    elif name in {"c4", "c4_stream"}:
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        for ex in ds:
            if ex.get("text", "").strip():
                yield ex["text"]
    elif name in {"ptb", "penn_treebank"}:
        # Priority 1: local file pre-downloaded by tools/download_ptb.py
        _local = Path(__file__).resolve().parent.parent / "data" / "ptb" / "ptb_test.txt"
        if _local.exists():
            for line in _local.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    yield line.strip()
            return
        # Priority 2: HF datasets (works only with datasets<3.0)
        _ptb_mirrors = [
            ("shenlong7/ptb_text_only", "penn_treebank"),
            ("FALcon6/ptb_text_only",   "penn_treebank"),
            ("ptb_text_only",           "penn_treebank"),
        ]
        ds = None
        for _repo, _cfg in _ptb_mirrors:
            try:
                ds = load_dataset(_repo, _cfg, split="test")
                break
            except Exception:
                continue
        if ds is None:
            print("  [warn] PTB local file not found and HF mirrors unavailable.")
            print("  [warn] Run: python tools/download_ptb.py")
            return
        for ex in ds:
            txt = ex.get("sentence", ex.get("text", ""))
            if txt.strip():
                yield txt
    else:
        raise ValueError(f"Unsupported PPL dataset: {dataset_name}")


@torch.no_grad()
def eval_ppl(model, tokenizer, datasets: list[str],
             seq_len: int, batch_size: int, device: str) -> dict[str, float]:
    model.eval()
    results: dict[str, float] = {}

    for ds_name in datasets:
        try:
            print(f"  loading {ds_name} ...", flush=True)
            eos = tokenizer.eos_token_id
            ids: list[int] = []
            for txt in _iter_texts(ds_name):
                ids.extend(tokenizer.encode(txt, add_special_tokens=False))
                if eos is not None:
                    ids.append(int(eos))
                if ds_name.lower() in {"c4", "c4_stream"} and len(ids) > 5_000_000:
                    break

            n_seq = (len(ids) - 1) // seq_len
            if n_seq == 0:
                print(f"  [warn] not enough tokens for {ds_name}, skipping")
                results[ds_name] = float("nan")
                continue

            flat = torch.tensor(ids[: n_seq * seq_len + 1], dtype=torch.long)
            x = flat[:-1].view(n_seq, seq_len)

            total_loss, total_tokens = 0.0, 0
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            for i in range(0, n_seq, batch_size):
                xb = x[i : i + batch_size].to(device)
                out = model(input_ids=xb, attention_mask=torch.ones_like(xb),
                            use_cache=False)
                loss = F.cross_entropy(
                    out.logits[:, :-1, :].contiguous().view(-1, out.logits.size(-1)),
                    xb[:, 1:].contiguous().view(-1),
                    reduction="sum",
                )
                total_loss   += float(loss.item())
                total_tokens += int(xb[:, 1:].numel())

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            ppl = math.exp(total_loss / total_tokens)
            results[ds_name] = ppl
            print(f"  {ds_name} PPL={ppl:.4f}  ({total_tokens} tokens, {dt:.1f}s)")

        except Exception as exc:
            print(f"  [error] {ds_name} PPL failed: {exc}")
            results[ds_name] = float("nan")

    return results


# ── lm-eval ───────────────────────────────────────────────────────────────────

_TASKS_DIR = Path(__file__).resolve().parent / "tasks"
_MATHQA_LOCAL = Path(__file__).resolve().parent.parent / "data" / "mathqa" / "test.jsonl"


def run_lmeval(model, tokenizer, tasks: list[str],
               batch_size: int | str,
               checkpoint: str | None = None,
               dtype_str: str = "bfloat16",
               device: str = "cuda:0") -> tuple[dict, str]:
    """Returns (raw_task_results, actual_batch_size_str, full_out).

    raw_task_results: the lm-eval out["results"] dict keyed by task name,
    each value is the full metric dict (acc,none / acc_norm,none / *,stderr …).

    If local data/mathqa/test.jsonl exists, 'mathqa' is swapped to the
    local task 'mathqa_local' (bypasses datasets script restriction).
    """
    # Replace 'mathqa' with local task if available
    use_mathqa_local = "mathqa" in tasks and _MATHQA_LOCAL.exists()
    if use_mathqa_local:
        tasks = ["mathqa_local" if t == "mathqa" else t for t in tasks]
        print(f"  [mathqa] using local file: {_MATHQA_LOCAL}")

    include_path = str(_TASKS_DIR) if use_mathqa_local else None

    print(f"\n--- lm-eval zero-shot: {tasks} ---", flush=True)

    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator as lm_evaluator

    use_path = checkpoint is not None and not (Path(checkpoint) / "model.pt").exists()
    if use_path:
        hflm = HFLM(pretrained=checkpoint, dtype=dtype_str,
                    batch_size=batch_size, device=device)
    else:
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    kwargs = dict(model=hflm, tasks=tasks, num_fewshot=0, log_samples=False)
    if include_path is not None:
        try:
            from lm_eval.tasks import TaskManager
            kwargs["task_manager"] = TaskManager(include_path=include_path)
        except Exception:
            kwargs["include_path"] = include_path
    out = lm_evaluator.simple_evaluate(**kwargs)
    raw = out["results"]

    # Remap mathqa_local → mathqa in results so callers see consistent key
    if use_mathqa_local and "mathqa_local" in raw:
        raw["mathqa"] = raw.pop("mathqa_local")

    actual_bs = str(getattr(hflm, "_batch_size",
                    getattr(hflm, "batch_size", batch_size)))
    print(f"  lm-eval batch_size: {actual_bs}")

    # Print summary (use original task names)
    orig_tasks = ["mathqa" if t == "mathqa_local" else t for t in tasks]
    for task in orig_tasks:
        if task not in raw:
            print(f"  [warn] '{task}' not in lm_eval output")
            continue
        primary = PRIMARY_METRIC.get(task, "acc")
        val = _get_metric(raw[task], primary)
        std = _get_stderr(raw[task], primary)
        print(f"  {task}: {val:.4f} ± {std:.4f}  ({primary})")

    return raw, actual_bs, out


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--model_family",  required=True,
                        help="Model family for grouping (e.g. llama, qwen).")
    parser.add_argument("--model_tag",     required=True,
                        help="Short name for CSV (e.g. Llama-3.1-8B).")
    parser.add_argument("--is_instruct",   required=True, choices=["0", "1"],
                        help="0 = base model, 1 = instruct model.")
    parser.add_argument("--method",        required=True)
    parser.add_argument("--keep_ratio",    required=True, type=float)
    parser.add_argument("--dtype",         default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device",        default="cuda:0")
    parser.add_argument("--output_csv",    default="results/decoder_eval.csv")
    parser.add_argument("--seq_len",       type=int, default=2048)
    parser.add_argument("--batch_size",       default="2",
                        help="Batch size for PPL eval.")
    parser.add_argument("--lmeval_batch_size", default="auto",
                        help="Batch size for lm-eval. 'auto' lets lm-eval maximize GPU usage.")
    parser.add_argument("--tasks",         default=DEFAULT_TASKS)
    parser.add_argument("--task_set",      default="main8",
                        help="Tag for the task set used (for result versioning).")
    parser.add_argument("--datasets",      default=DEFAULT_DATASETS)
    parser.add_argument("--hf_token",      default="")
    parser.add_argument("--tokenizer",     default="",
                        help="Tokenizer path/id override (for model.pt checkpoints "
                             "whose directory lacks tokenizer files).")
    parser.add_argument("--no_ppl",        action="store_true")
    parser.add_argument("--no_lmeval",     action="store_true")
    parser.add_argument("--compression_success", default="yes",
                        choices=["yes", "no"])
    parser.add_argument("--notes",         default="")
    parser.add_argument("--force",         action="store_true",
                        help="Re-run even if this entry already exists in the CSV.")
    args = parser.parse_args()

    if not args.force and _already_done(args.output_csv, args.model_tag,
                                        args.method, args.keep_ratio):
        print(f"[skip] already in CSV: {args.model_tag} {args.method} keep={args.keep_ratio}")
        return

    dtype = {"bf16": torch.bfloat16,
             "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    ckpt_size_gb = _dir_size_gb(args.checkpoint)

    # ── load ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"method={args.method}  keep_ratio={args.keep_ratio}  dtype={args.dtype}")
    print(f"checkpoint: {args.checkpoint}")
    t0 = time.time()
    model, tokenizer = load_model(args.checkpoint, dtype, args.device,
                                  hf_token=args.hf_token or None,
                                  tokenizer_path=args.tokenizer or None)
    model.eval()
    print(f"loaded in {time.time() - t0:.1f}s")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # ── PPL ───────────────────────────────────────────────────────────────────
    ppl: dict[str, float] = {}
    ppl_ok = True
    if not args.no_ppl:
        print("\n--- PPL ---")
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        bs_ppl   = int(args.batch_size) if args.batch_size != "auto" else 1
        try:
            ppl = eval_ppl(model, tokenizer, datasets,
                           args.seq_len, bs_ppl, args.device)
        except Exception as exc:
            print(f"[error] PPL failed: {exc}")
            ppl_ok = False

    # ── lm-eval ───────────────────────────────────────────────────────────────
    lmeval_raw: dict = {}       # task → full metric dict
    lmeval_full_out: dict = {}  # full lm-eval output (for JSON)
    lmeval_ok = True
    actual_lmeval_bs = args.lmeval_batch_size
    if not args.no_lmeval:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        try:
            lmeval_raw, actual_lmeval_bs, lmeval_full_out = run_lmeval(
                model, tokenizer, tasks, args.lmeval_batch_size,
                checkpoint=args.checkpoint,
                dtype_str={"bf16": "bfloat16", "fp16": "float16",
                           "fp32": "float32"}[args.dtype],
                device=args.device)
        except Exception as exc:
            print(f"[error] lm-eval failed: {exc}")
            lmeval_ok = False

    eval_success = "yes" if (ppl_ok and lmeval_ok) else "no"

    # ── save lm-eval JSON ─────────────────────────────────────────────────────
    out_path = Path(args.output_csv)
    if lmeval_full_out:
        json_dir = out_path.parent / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = args.model_tag.replace("/", "_").replace(" ", "_")
        ts_compact = timestamp.replace(":", "").replace("-", "").replace("T", "_")
        json_name  = f"{safe_tag}_{args.method}_{args.keep_ratio}_{ts_compact}.json"
        json_path  = json_dir / json_name
        try:
            # Save only aggregated results (not per-sample logs) to keep size small
            with open(json_path, "w") as jf:
                json.dump(lmeval_full_out.get("results", lmeval_full_out),
                          jf, indent=2, default=str)
            print(f"  lm-eval JSON → {json_path}")
        except Exception as exc:
            print(f"  [warn] could not save JSON: {exc}")

    # ── avg_score (primary metric per task) ───────────────────────────────────
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    valid: list[float] = []
    for task in task_list:
        primary = PRIMARY_METRIC.get(task, "acc")
        v = _get_metric(lmeval_raw.get(task, {}), primary)
        if not math.isnan(v):
            valid.append(v)
    avg_score = sum(valid) / len(valid) if valid else float("nan")

    # ── build CSV row ─────────────────────────────────────────────────────────
    row: dict = {
        "model_family":        args.model_family,
        "model_tag":           args.model_tag,
        "is_instruct":         args.is_instruct,
        "method":              args.method,
        "keep_ratio":          args.keep_ratio,
        "dtype":               args.dtype,
        "compression_success": args.compression_success,
        "eval_success":        eval_success,
        "task_set":            args.task_set,
        "metric_protocol":     METRIC_PROTOCOL,
        "wikitext2_ppl":       _fmt(ppl.get("wikitext2", float("nan"))),
        "c4_ppl":              _fmt(ppl.get("c4_stream", ppl.get("c4", float("nan")))),
        "ptb_ppl":             _fmt(ppl.get("ptb",       float("nan"))),
        "avg_score":           _fmt(avg_score),
        "checkpoint_path":     args.checkpoint,
        "checkpoint_size_gb":  _fmt(ckpt_size_gb),
        "seq_len":             args.seq_len,
        "eval_batch_size":     actual_lmeval_bs,
        "device":              args.device,
        "lm_eval_version":     _lm_eval_version(),
        "transformers_version": _transformers_version(),
        "timestamp":           timestamp,
        "notes":               args.notes,
    }

    # Per-task columns
    for task in ORDERED_TASKS:
        tr = lmeval_raw.get(task, {})
        row[f"{task}_acc"]      = _fmt(_get_metric(tr, "acc"))
        row[f"{task}_acc_norm"] = _fmt(_get_metric(tr, "acc_norm"))
        primary = PRIMARY_METRIC.get(task, "acc")
        row[f"{task}_std"]      = _fmt(_get_stderr(tr, primary))

    # ── write CSV ─────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n{'='*64}")
    print(f"Done → {out_path}")
    print(f"  wiki2={row['wikitext2_ppl']}  c4={row['c4_ppl']}  "
          f"avg={row['avg_score']}  eval_success={eval_success}")


if __name__ == "__main__":
    main()
