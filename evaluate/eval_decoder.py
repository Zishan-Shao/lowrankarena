"""
eval_decoder.py — unified decoder LLM evaluation.

Input:  one HF checkpoint directory (or HF model id for baseline)
Output: one appended row in a CSV

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
import math
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

# ── constants ─────────────────────────────────────────────────────────────────

TASK_METRICS: dict[str, str] = {
    "piqa":          "acc",
    "hellaswag":     "acc_norm",
    "arc_easy":      "acc",
    "arc_challenge": "acc_norm",
    "winogrande":    "acc",
    "openbookqa":    "acc_norm",
}
DEFAULT_TASKS    = ",".join(TASK_METRICS)
DEFAULT_DATASETS = "wikitext2,c4"

METRIC_PROTOCOL = ";".join(f"{t}={m}" for t, m in TASK_METRICS.items())

CSV_FIELDS = [
    # identity
    "model_family", "model_tag", "is_instruct", "method", "keep_ratio", "dtype",
    # status
    "compression_success", "eval_success",
    # protocol
    "task_set", "metric_protocol",
    # PPL
    "wikitext2_ppl", "c4_ppl", "ptb_ppl",
    # zero-shot tasks
    "piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "openbookqa",
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
    """Load a model from a .pt file inside an otherwise HF-style directory.

    Dobi-SVD saves SVDTransformLayer objects, so we register those classes
    before calling torch.load to avoid unpickling errors.
    """
    dobi_root = Path(__file__).resolve().parent.parent / "baselines" / "Dobi-SVD"
    for p in (str(dobi_root), str(dobi_root / "modules")):
        if p not in __import__("sys").path:
            __import__("sys").path.insert(0, p)
    try:
        import modules.module as _  # noqa: F401
    except ImportError:
        try:
            import module as _  # noqa: F401
        except ImportError:
            pass  # no SVDTransformLayer needed if not Dobi

    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    # obj may be the model directly, or {'model': ..., 'tokenizer': ...}
    model = obj["model"] if isinstance(obj, dict) else obj
    return model.to(dtype=dtype).to(device)


def load_model(checkpoint: str, dtype: torch.dtype, device: str,
               hf_token: str | None):
    ckpt_dir = Path(checkpoint)
    model_pt = ckpt_dir / "model.pt"

    # Directory has model.pt instead of standard safetensors/bin weights
    if model_pt.is_file():
        from transformers import AutoTokenizer
        extra = {"token": hf_token} if hf_token else {}
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True, **extra
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
    elif name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        for ex in ds:
            if ex.get("text", "").strip():
                yield ex["text"]
    elif name in {"ptb", "penn_treebank"}:
        ds = load_dataset("FALcon6/ptb_text_only", split="test")
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
                if ds_name == "c4" and len(ids) > 5_000_000:
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

def run_lmeval(model, tokenizer, tasks: list[str],
               batch_size: int | str,
               checkpoint: str | None = None,
               dtype_str: str = "bfloat16",
               device: str = "cuda:0") -> dict[str, float]:
    print(f"\n--- lm-eval zero-shot: {tasks} ---", flush=True)

    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator as lm_evaluator

    # Prefer passing the checkpoint path string so lm-eval loads the model
    # itself — this is the documented HF usage and avoids the "pretrained is
    # not str" warning.  Fall back to passing the model object when the
    # checkpoint is not a standard HF dir (e.g. Dobi's model.pt layout).
    use_path = (
        checkpoint is not None
        and Path(checkpoint).is_dir()
        and not (Path(checkpoint) / "model.pt").exists()
    )

    if use_path:
        hflm = HFLM(
            pretrained=checkpoint,
            dtype=dtype_str,
            batch_size=batch_size,
            device=device,
        )
    else:
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    out = lm_evaluator.simple_evaluate(model=hflm, tasks=tasks, num_fewshot=0)
    raw = out["results"]

    scores: dict[str, float] = {}
    for task in tasks:
        if task not in raw:
            print(f"  [warn] '{task}' not in lm_eval output")
            scores[task] = float("nan")
            continue
        metric = TASK_METRICS.get(task, "acc")
        scores[task] = _get_metric(raw[task], metric)
        print(f"  {task}: {scores[task]:.4f}  ({metric})")
    return scores


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
    parser.add_argument("--batch_size",    default="2")
    parser.add_argument("--tasks",         default=DEFAULT_TASKS)
    parser.add_argument("--task_set",      default="main6",
                        help="Tag for the task set used (for result versioning).")
    parser.add_argument("--datasets",      default=DEFAULT_DATASETS)
    parser.add_argument("--hf_token",      default="")
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
                                  hf_token=args.hf_token or None)
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
    lmeval: dict[str, float] = {}
    lmeval_ok = True
    if not args.no_lmeval:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        try:
            lmeval = run_lmeval(model, tokenizer, tasks, args.batch_size,
                                checkpoint=args.checkpoint,
                                dtype_str={"bf16": "bfloat16", "fp16": "float16",
                                           "fp32": "float32"}[args.dtype],
                                device=args.device)
        except Exception as exc:
            print(f"[error] lm-eval failed: {exc}")
            lmeval_ok = False
            for t in tasks:
                lmeval[t] = float("nan")

    eval_success = "yes" if (ppl_ok and lmeval_ok) else "no"

    # ── avg_score ─────────────────────────────────────────────────────────────
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    valid     = [lmeval[t] for t in task_list
                 if not math.isnan(lmeval.get(t, float("nan")))]
    avg_score = sum(valid) / len(valid) if valid else float("nan")

    # ── write CSV ─────────────────────────────────────────────────────────────
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
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
        "c4_ppl":              _fmt(ppl.get("c4",        float("nan"))),
        "ptb_ppl":             _fmt(ppl.get("ptb",       float("nan"))),
        "piqa":                _fmt(lmeval.get("piqa",          float("nan"))),
        "hellaswag":           _fmt(lmeval.get("hellaswag",     float("nan"))),
        "arc_easy":            _fmt(lmeval.get("arc_easy",      float("nan"))),
        "arc_challenge":       _fmt(lmeval.get("arc_challenge", float("nan"))),
        "winogrande":          _fmt(lmeval.get("winogrande",    float("nan"))),
        "openbookqa":          _fmt(lmeval.get("openbookqa",    float("nan"))),
        "avg_score":           _fmt(avg_score),
        "checkpoint_path":     args.checkpoint,
        "checkpoint_size_gb":  _fmt(ckpt_size_gb),
        "seq_len":             args.seq_len,
        "eval_batch_size":     args.batch_size,
        "device":              args.device,
        "lm_eval_version":     _lm_eval_version(),
        "transformers_version": _transformers_version(),
        "timestamp":           timestamp,
        "notes":               args.notes,
    }

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
