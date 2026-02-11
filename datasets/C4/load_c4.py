"""
Quick loader for allenai/c4 variants, with safe small-slice defaults.

Examples:
  # load 10 examples from validation split (non-streaming)
  python datasets/C4/load_c4.py --variant en --split validation[:10]

  # streaming: read a few examples from train
  python datasets/C4/load_c4.py --variant en --split train --stream
"""

from datasets import load_dataset
import argparse
import itertools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', type=str, default='en', help='C4 config (en, en.noclean, en.noblocklist, realnewslike, multilingual)')
    ap.add_argument('--split', type=str, default='validation[:10]', help='Dataset split or slice, e.g., validation[:10], train, train[:1000]')
    ap.add_argument('--stream', action='store_true', help='Use streaming mode (don\'t download full shards).')
    args = ap.parse_args()

    ds = load_dataset('allenai/c4', args.variant, split=args.split, streaming=args.stream)
    if args.stream:
        it = iter(ds)
        sample = list(itertools.islice(it, 3))
        print(f"streamed {len(sample)} examples; keys: {list(sample[0].keys()) if sample else []}")
    else:
        print(f"loaded {len(ds)} examples; keys: {ds[0].keys() if len(ds) else []}")


if __name__ == '__main__':
    main()
