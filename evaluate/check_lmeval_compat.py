"""
check_lmeval_compat.py — quickly verify that a checkpoint works with lm-eval's
loglikelihood interface and is not outputting degenerate (constant) logits.

Usage:
    python evaluate/check_lmeval_compat.py \
        --checkpoint ../hf_ckpts/LowRankArena/llama31_8b/SVDLLMv1/hf_whitening_then_update_0.8 \
        --tokenizer  meta-llama/Llama-3.1-8B \
        --device cuda:0

    # For model.pt checkpoints, also pass --tokenizer:
    python evaluate/check_lmeval_compat.py \
        --checkpoint ../hf_ckpts/.../model_dir \
        --tokenizer  meta-llama/Llama-3.1-8B
"""
import argparse
import sys
import math
from pathlib import Path

import torch

# ── allow importing load_model from eval_decoder ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_decoder import load_model, _resolve_checkpoint

# ── BoolQ-style test pairs (context, yes_continuation, no_continuation) ──────
# Each tuple: (premise, "yes", "no")  — same format lm-eval uses for BoolQ
_BOOLQ_SAMPLES = [
    ("Passage: Persian is the official language of Iran.\nQuestion: is persian the official language of iran?", " yes", " no"),
    ("Passage: The sun rises in the east.\nQuestion: does the sun rise in the west?", " yes", " no"),
    ("Passage: Water boils at 100 degrees Celsius at sea level.\nQuestion: does water boil at 100 degrees celsius?", " yes", " no"),
    ("Passage: Gold is a metal.\nQuestion: is gold a type of plastic?", " yes", " no"),
    ("Passage: Cats are mammals.\nQuestion: are cats reptiles?", " yes", " no"),
]

# ── MathQA-style test pairs ───────────────────────────────────────────────────
_MATHQA_SAMPLES = [
    ("Question: What is 2 + 2?\nOptions: a ) 3 , b ) 4 , c ) 5 , d ) 6 , e ) 7\nAnswer:", " a", " b", " c", " d", " e"),
    ("Question: What is 10 - 3?\nOptions: a ) 5 , b ) 6 , c ) 7 , d ) 8 , e ) 9\nAnswer:", " a", " b", " c", " d", " e"),
    ("Question: What is 3 * 4?\nOptions: a ) 10 , b ) 11 , c ) 12 , d ) 13 , e ) 14\nAnswer:", " a", " b", " c", " d", " e"),
]

# ── correct answers (0-indexed choice) ───────────────────────────────────────
_BOOLQ_CORRECT  = [0, 1, 0, 1, 1]   # yes=0, no=1
_MATHQA_CORRECT = [1, 2, 2]          # b=1, c=2, c=2


@torch.no_grad()
def _loglikelihood_direct(model, tokenizer, context: str, continuations: list[str],
                          device: str, use_cache: bool = False) -> list[float]:
    """Compute log P(continuation | context) for each continuation via direct
    forward pass.  use_cache=False mirrors PPL eval; True mirrors lm-eval default."""
    scores = []
    ctx_ids = tokenizer.encode(context, add_special_tokens=True)
    for cont in continuations:
        cont_ids = tokenizer.encode(cont, add_special_tokens=False)
        full_ids = ctx_ids + cont_ids
        inp = torch.tensor([full_ids], dtype=torch.long, device=device)
        out = model(input_ids=inp, use_cache=use_cache)
        logits = out.logits[0]  # [T, V]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # sum log probs over the continuation tokens
        score = 0.0
        for i, tok in enumerate(cont_ids):
            pos = len(ctx_ids) - 1 + i  # position that predicts tok
            score += float(log_probs[pos, tok].item())
        scores.append(score)
    return scores


def _loglikelihood_hflm(model, tokenizer, context: str, continuations: list[str],
                        device: str) -> list[float]:
    """Compute log P(continuation | context) via lm-eval's HFLM interface."""
    from lm_eval.models.huggingface import HFLM
    from lm_eval.api.instance import Instance

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

    requests = [
        Instance(request_type="loglikelihood",
                 doc={}, idx=i,
                 arguments=(context, cont))
        for i, cont in enumerate(continuations)
    ]
    results = hflm.loglikelihood(requests)
    return [r[0] for r in results]   # r = (log_prob, is_greedy)


def _evaluate_samples(model, tokenizer, device, use_hflm: bool,
                      use_cache_override: bool = False):
    print(f"\n{'─'*60}")
    mode = "HFLM (lm-eval)" if use_hflm else f"Direct forward (use_cache={use_cache_override})"
    print(f"Mode: {mode}")
    print(f"{'─'*60}")

    if use_hflm:
        fn = _loglikelihood_hflm
    else:
        fn = lambda model, tokenizer, ctx, conts, dev: _loglikelihood_direct(
            model, tokenizer, ctx, conts, dev, use_cache=use_cache_override)

    # ── BoolQ ─────────────────────────────────────────────────────────────────
    print("\n[BoolQ]  correct=yes(0) when true, no(1) when false")
    boolq_preds, boolq_correct = [], 0
    for i, (ctx, yes, no) in enumerate(_BOOLQ_SAMPLES):
        scores = fn(model, tokenizer, ctx, [yes, no], device)
        pred = int(scores[1] > scores[0])   # 0=yes, 1=no
        correct = _BOOLQ_CORRECT[i]
        mark = "✓" if pred == correct else "✗"
        print(f"  ex{i}: log_yes={scores[0]:+.3f}  log_no={scores[1]:+.3f}"
              f"  → {'no' if pred else 'yes':3s}  {mark}")
        boolq_preds.append(pred)
        if pred == correct:
            boolq_correct += 1

    all_same_boolq = len(set(boolq_preds)) == 1
    print(f"  acc={boolq_correct}/{len(_BOOLQ_SAMPLES)}"
          f"  all_same={'YES ← DEGENERATE' if all_same_boolq else 'no'}")

    # ── MathQA ────────────────────────────────────────────────────────────────
    print("\n[MathQA]  choices: a b c d e")
    mathqa_preds, mathqa_correct = [], 0
    choice_labels = list(" abcde")
    for i, (ctx, *conts) in enumerate(_MATHQA_SAMPLES):
        scores = fn(model, tokenizer, ctx, conts, device)
        pred = int(torch.tensor(scores).argmax())
        correct = _MATHQA_CORRECT[i]
        mark = "✓" if pred == correct else "✗"
        score_str = "  ".join(f"{choice_labels[j+1]}:{scores[j]:+.3f}"
                              for j in range(len(conts)))
        print(f"  ex{i}: {score_str}  → {choice_labels[pred+1]}  {mark}")
        mathqa_preds.append(pred)
        if pred == correct:
            mathqa_correct += 1

    all_same_mathqa = len(set(mathqa_preds)) == 1
    print(f"  acc={mathqa_correct}/{len(_MATHQA_SAMPLES)}"
          f"  all_same={'YES ← DEGENERATE' if all_same_mathqa else 'no'}")

    return all_same_boolq, all_same_mathqa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer",  default="",
                        help="Tokenizer path if checkpoint dir has no tokenizer files.")
    parser.add_argument("--dtype",  default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_hflm", action="store_true",
                        help="Skip the lm-eval HFLM test (faster).")
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]

    ckpt = _resolve_checkpoint(args.checkpoint)
    print(f"Loading: {ckpt}")
    model, tokenizer = load_model(ckpt, dtype, args.device,
                                  hf_token=None,
                                  tokenizer_path=args.tokenizer or None)
    model.eval()
    print(f"Loaded: {type(model).__name__}")

    # ── Direct forward, use_cache=False (same as PPL eval) ───────────────────
    print("\n[use_cache=False — same path as PPL eval]")
    deg_boolq_d, deg_mathqa_d = _evaluate_samples(model, tokenizer,
                                                   args.device, use_hflm=False)

    # ── Direct forward, use_cache=True (model default, same as lm-eval) ──────
    print("\n[use_cache=True — model default, mirrors lm-eval calling convention]")
    deg_boolq_c, deg_mathqa_c = _evaluate_samples(model, tokenizer,
                                                   args.device, use_hflm=False,
                                                   use_cache_override=True)
    if (deg_boolq_d != deg_boolq_c) or (deg_mathqa_d != deg_mathqa_c):
        print("\n  !! use_cache=True changes results → KV-cache path is broken in this checkpoint")

    # ── lm-eval HFLM ─────────────────────────────────────────────────────────
    if not args.no_hflm:
        try:
            deg_boolq_h, deg_mathqa_h = _evaluate_samples(model, tokenizer,
                                                           args.device, use_hflm=True)
        except ImportError:
            print("\n[skip] lm_eval not installed; skipping HFLM test.")
            deg_boolq_h = deg_mathqa_h = None

        # ── compare ──────────────────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print("Summary")
        print(f"  use_cache=False (PPL path): boolq_deg={deg_boolq_d}  mathqa_deg={deg_mathqa_d}")
        print(f"  use_cache=True  (lm-eval default): boolq_deg={deg_boolq_c}  mathqa_deg={deg_mathqa_c}")
        if deg_boolq_h is not None:
            print(f"  HFLM (actual lm-eval):  boolq_deg={deg_boolq_h}  mathqa_deg={deg_mathqa_h}")
            if (deg_boolq_d != deg_boolq_h) or (deg_mathqa_d != deg_mathqa_h):
                print("  !! MISMATCH: lm-eval gives different results than direct(use_cache=False)"
                      " → checkpoint has a KV-cache or calling-convention bug")

    print()


if __name__ == "__main__":
    main()
