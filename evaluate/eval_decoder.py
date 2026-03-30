"""
eval_decoder.py — unified decoder LLM evaluation.

Input:  one HF checkpoint directory (or HF model id for baseline)
Output: one appended row in a CSV

Usage:
    python eval_decoder.py \
        --checkpoint /path/to/hf_dir \
        --model_tag  Llama-3.1-8B \
        --method     SVDLLMv2 \
        --keep_ratio 0.8 \
        --dtype bf16 --device cuda:0 \
        --output_csv results/llama31_8b.csv

    # Baseline from HF Hub:
    python eval_decoder.py \
        --checkpoint meta-llama/Llama-3.1-8B \
        --model_tag Llama-3.1-8B \
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

CSV_FIELDS = [
    "model_tag", "method", "keep_ratio", "dtype",
    "wikitext2_ppl", "c4_ppl", "ptb_ppl",
    "piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "openbookqa",
    "avg_score",
    "checkpoint_path", "notes",
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


# ── loading ───────────────────────────────────────────────────────────────────

def load_model(checkpoint: str, dtype: torch.dtype, device: str,
               hf_token: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    extra = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True, **extra
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
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
    elif name in {"c4"}:
        ds = load_dataset("c4", "en", split="validation", streaming=True)
        for ex in ds:
            if ex.get("text", "").strip():
                yield ex["text"]
    elif name in {"ptb", "penn_treebank"}:
        ds = load_dataset("ptb_text_only", split="test")
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
        print(f"  loading {ds_name} ...", flush=True)
        eos = tokenizer.eos_token_id
        ids: list[int] = []
        for txt in _iter_texts(ds_name):
            ids.extend(tokenizer.encode(txt, add_special_tokens=False))
            if eos is not None:
                ids.append(int(eos))
            # c4 is streaming; 5 M tokens is enough for a stable PPL estimate
            if ds_name == "c4" and len(ids) > 5_000_000:
                break

        n_seq = (len(ids) - 1) // seq_len
        if n_seq == 0:
            print(f"  [warn] not enough tokens for {ds_name}, skipping")
            results[ds_name] = float("nan")
            continue

        flat = torch.tensor(ids[: n_seq * seq_len + 1], dtype=torch.long)
        x = flat[:-1].view(n_seq, seq_len)

        total_loss = 0.0
        total_tokens = 0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for i in range(0, n_seq, batch_size):
            xb = x[i : i + batch_size].to(device)
            out = model(input_ids=xb, attention_mask=torch.ones_like(xb),
                        use_cache=False)
            logits = out.logits                       # [B, S, V]
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
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

    return results


# ── lm-eval ───────────────────────────────────────────────────────────────────

def run_lmeval(model, tokenizer, tasks: list[str],
               batch_size: int | str) -> dict[str, float]:
    print(f"\n--- lm-eval zero-shot: {tasks} ---", flush=True)

    try:
        from lm_eval.models.huggingface import HFLM
        from lm_eval import evaluator as lm_evaluator

        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        out  = lm_evaluator.simple_evaluate(model=hflm, tasks=tasks, num_fewshot=0)
        raw  = out["results"]

    except (ImportError, TypeError):
        # lm_eval v0.3 fallback
        from lm_eval.base import BaseLM
        from lm_eval import evaluator as lm_evaluator

        class _LM(BaseLM):
            def __init__(self, m, tok, b):
                super().__init__()
                self._m, self._tok = m, tok
                self._bs = int(b) if str(b).isdigit() else 1
                self._dev = next(m.parameters()).device

            @property
            def eot_token_id(self): return self._tok.eos_token_id
            @property
            def max_length(self):
                return getattr(self._m.config, "max_position_embeddings", 2048)
            @property
            def max_gen_toks(self): return 256
            @property
            def batch_size(self): return self._bs
            @property
            def device(self): return self._dev
            def tok_encode(self, s): return self._tok.encode(s, add_special_tokens=False)
            def tok_decode(self, ts): return self._tok.decode(ts)
            def _model_call(self, inps):
                with torch.no_grad():
                    return self._m(inps)[0]
            def _model_generate(self, ctx, max_len, eos):
                return self._m.generate(ctx, max_length=max_len,
                                        eos_token_id=eos, do_sample=False)

        lm_obj = _LM(model, tokenizer, batch_size)
        out = lm_evaluator.simple_evaluate(
            lm_obj, tasks=tasks, num_fewshot=0, no_cache=True
        )
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
    parser.add_argument("--checkpoint",  required=True,
                        help="HF model id or local HF dir.")
    parser.add_argument("--model_tag",   required=True,
                        help="Short name written to CSV (e.g. Llama-3.1-8B).")
    parser.add_argument("--method",      required=True,
                        help="Compression method name for CSV.")
    parser.add_argument("--keep_ratio",  required=True, type=float,
                        help="1 - compression_ratio; 1.0 for baseline.")
    parser.add_argument("--dtype",       default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--output_csv",  default="results/decoder_eval.csv")
    parser.add_argument("--seq_len",     type=int, default=2048)
    parser.add_argument("--batch_size",  default="2",
                        help="Batch size for PPL and lm-eval. 'auto' lets lm-eval choose.")
    parser.add_argument("--tasks",       default=DEFAULT_TASKS)
    parser.add_argument("--datasets",    default=DEFAULT_DATASETS,
                        help="Comma-separated PPL datasets: wikitext2, c4, ptb.")
    parser.add_argument("--hf_token",    default="")
    parser.add_argument("--no_ppl",      action="store_true")
    parser.add_argument("--no_lmeval",   action="store_true")
    parser.add_argument("--notes",       default="")
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16,
             "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]

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
    if not args.no_ppl:
        print("\n--- PPL ---")
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        bs_ppl   = int(args.batch_size) if args.batch_size != "auto" else 1
        ppl = eval_ppl(model, tokenizer, datasets,
                       args.seq_len, bs_ppl, args.device)

    # ── lm-eval ───────────────────────────────────────────────────────────────
    lmeval: dict[str, float] = {}
    if not args.no_lmeval:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        try:
            lmeval = run_lmeval(model, tokenizer, tasks, args.batch_size)
        except Exception as exc:
            print(f"[error] lm-eval failed: {exc}")
            for t in tasks:
                lmeval[t] = float("nan")

    # ── peak memory ───────────────────────────────────────────────────────────
    peak_gb = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # ── avg_score ─────────────────────────────────────────────────────────────
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    valid     = [lmeval[t] for t in task_list
                 if not math.isnan(lmeval.get(t, float("nan")))]
    avg_score = sum(valid) / len(valid) if valid else float("nan")

    # ── write CSV ─────────────────────────────────────────────────────────────
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "model_tag":      args.model_tag,
        "method":         args.method,
        "keep_ratio":     args.keep_ratio,
        "dtype":          args.dtype,
        "wikitext2_ppl":  _fmt(ppl.get("wikitext2", float("nan"))),
        "c4_ppl":         _fmt(ppl.get("c4",        float("nan"))),
        "ptb_ppl":        _fmt(ppl.get("ptb",       float("nan"))),
        "piqa":           _fmt(lmeval.get("piqa",          float("nan"))),
        "hellaswag":      _fmt(lmeval.get("hellaswag",     float("nan"))),
        "arc_easy":       _fmt(lmeval.get("arc_easy",      float("nan"))),
        "arc_challenge":  _fmt(lmeval.get("arc_challenge", float("nan"))),
        "winogrande":     _fmt(lmeval.get("winogrande",    float("nan"))),
        "openbookqa":     _fmt(lmeval.get("openbookqa",    float("nan"))),
        "avg_score":      _fmt(avg_score),
        "checkpoint_path": args.checkpoint,
        "notes":          args.notes,
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
          f"avg={row['avg_score']}  peak={peak_gb:.1f}GB")


if __name__ == "__main__":
    main()
