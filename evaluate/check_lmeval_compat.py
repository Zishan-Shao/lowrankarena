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
                          device: str) -> list[float]:
    """Compute log P(continuation | context) for each continuation via direct
    forward pass (same teacher-forcing as PPL eval)."""
    scores = []
    ctx_ids = tokenizer.encode(context, add_special_tokens=True)
    for cont in continuations:
        cont_ids = tokenizer.encode(cont, add_special_tokens=False)
        full_ids = ctx_ids + cont_ids
        inp = torch.tensor([full_ids], dtype=torch.long, device=device)
        out = model(input_ids=inp, use_cache=False)
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


def _evaluate_samples(model, tokenizer, device, use_hflm: bool):
    print(f"\n{'─'*60}")
    mode = "HFLM (lm-eval)" if use_hflm else "Direct forward"
    print(f"Mode: {mode}")
    print(f"{'─'*60}")

    fn = _loglikelihood_hflm if use_hflm else _loglikelihood_direct

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

    # ── Direct forward (same as PPL eval) ────────────────────────────────────
    deg_boolq_d, deg_mathqa_d = _evaluate_samples(model, tokenizer,
                                                   args.device, use_hflm=False)

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
        print("Comparison: Direct vs HFLM")
        if deg_boolq_h is not None:
            boolq_match  = deg_boolq_d  == deg_boolq_h
            mathqa_match = deg_mathqa_d == deg_mathqa_h
            print(f"  BoolQ  degenerate: direct={deg_boolq_d}  hflm={deg_boolq_h}"
                  f"  match={boolq_match}")
            print(f"  MathQA degenerate: direct={deg_mathqa_d}  hflm={deg_mathqa_h}"
                  f"  match={mathqa_match}")
            if not boolq_match or not mathqa_match:
                print("  !! MISMATCH: lm-eval HFLM gives different results than "
                      "direct forward → lm-eval compat bug in this checkpoint")

    print()


if __name__ == "__main__":
    main()
