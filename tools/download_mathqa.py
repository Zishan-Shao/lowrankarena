#!/usr/bin/env python3
"""
Download mathqa test split (parquet version) and save to data/mathqa/test.jsonl.

Uses parquet mirror to bypass datasets>=3.0 script restriction.

Usage (run from lowrankarena/):
    python tools/download_mathqa.py
    python tools/download_mathqa.py --out data/mathqa/test.jsonl
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mathqa/test.jsonl")
    args = ap.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent.parent / out_path

    if out_path.exists():
        n = sum(1 for _ in out_path.open())
        print(f"Already exists: {out_path}  ({n} examples)")
        return

    from datasets import load_dataset

    mirrors = [
        ("regisss/math_qa",  "refs/convert/parquet"),
        ("swiss-ai/math_qa", "refs/convert/parquet"),
    ]

    ds = None
    for repo, revision in mirrors:
        try:
            print(f"  trying {repo} @ {revision} ...")
            ds = load_dataset(repo, revision=revision, split="test")
            print(f"  loaded {len(ds)} examples from {repo}")
            break
        except Exception as e:
            print(f"    failed: {e}")

    if ds is None:
        print("\n[ERROR] All mirrors failed.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(ds)} examples → {out_path}")


if __name__ == "__main__":
    main()
