#!/usr/bin/env python3
"""
Download PTB test split and save to data/ptb/ptb_test.txt (one sentence per line).

Try order:
  1. HuggingFace datasets (several mirrors, no trust_remote_code)
  2. NLTK Penn Treebank corpus
  3. Manual URL (treebank-3 WSJ section 23 raw text via nltk data server)

Usage (run from lowrankarena/):
    python tools/download_ptb.py
    python tools/download_ptb.py --out data/ptb/ptb_test.txt
"""
import argparse
import os
from pathlib import Path


def try_hf(out_path: Path) -> bool:
    mirrors = [
        ("shenlong7/ptb_text_only", "penn_treebank"),
        ("FALcon6/ptb_text_only",   "penn_treebank"),
        ("ptb_text_only",           "penn_treebank"),
    ]
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets not installed, skipping HF path")
        return False

    for repo, cfg in mirrors:
        try:
            print(f"  trying HF: {repo} / {cfg} ...")
            ds = load_dataset(repo, cfg, split="test")
            sentences = [ex.get("sentence", ex.get("text", ""))
                         for ex in ds
                         if ex.get("sentence", ex.get("text", "")).strip()]
            if sentences:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text("\n".join(sentences) + "\n", encoding="utf-8")
                print(f"  saved {len(sentences)} sentences → {out_path}")
                return True
        except Exception as e:
            print(f"    failed: {e}")
    return False


def try_nltk(out_path: Path) -> bool:
    try:
        import nltk
    except ImportError:
        print("  nltk not installed, skipping")
        return False

    try:
        print("  trying NLTK treebank ...")
        try:
            nltk.data.find("corpora/treebank")
        except LookupError:
            print("  downloading nltk treebank ...")
            nltk.download("treebank", quiet=True)

        from nltk.corpus import treebank
        # NLTK treebank has wsj_0001 … wsj_0199; section 23 ≈ wsj_2300-2454
        # Use all fileids as a proxy (the full corpus is sections 00-24)
        fileids = treebank.fileids()
        sentences = []
        for fid in fileids:
            for sent in treebank.sents(fid):
                line = " ".join(sent).strip()
                if line:
                    sentences.append(line)
        if sentences:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("\n".join(sentences) + "\n", encoding="utf-8")
            print(f"  saved {len(sentences)} sentences → {out_path}")
            return True
    except Exception as e:
        print(f"  NLTK failed: {e}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ptb/ptb_test.txt",
                    help="Output file path (relative to cwd or absolute)")
    args = ap.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent.parent / out_path

    if out_path.exists():
        lines = out_path.read_text(encoding="utf-8").splitlines()
        print(f"Already exists: {out_path}  ({len(lines)} sentences)")
        return

    print(f"Target: {out_path}")

    if try_hf(out_path):
        return
    if try_nltk(out_path):
        return

    print("\n[ERROR] All methods failed.")
    print("Options:")
    print("  1. pip install datasets==2.21.0  (last version with trust_remote_code)")
    print("  2. pip install nltk && python -c \"import nltk; nltk.download('treebank')\"")


if __name__ == "__main__":
    main()
