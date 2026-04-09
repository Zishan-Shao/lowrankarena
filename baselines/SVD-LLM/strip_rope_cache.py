"""
strip_rope_cache.py
--------------------
Post-process existing SVDLLM .pt checkpoints to remove pre-computed
cos_cached / sin_cached buffers from every LlamaRotaryEmbedding.

These buffers are marked persistent=False (not needed for correctness)
but are included in torch.save because it pickles the full object.
Stripping them saves ~4 GB per checkpoint (fp32 tables for 32 layers).

forward() in flashsvd_component/svd_llama.py already handles None
cos_cached via lazy recompute, so stripped checkpoints work normally.

Usage:
    python strip_rope_cache.py checkpoints/svdllm/llama31_8b/
    python strip_rope_cache.py checkpoints/svdllm/llama31_8b/meta_llama_Llama_3.1_8B_whitening_only_0.8.pt
    python strip_rope_cache.py checkpoints/svdllm/  --recursive
"""

import argparse
import os
import sys
from pathlib import Path

import torch


def strip_rope_cache(model):
    """Set cos_cached / sin_cached to None for every rotary_emb in model."""
    count = 0
    for m in model.modules():
        re = getattr(m, 'rotary_emb', None)
        if re is None:
            continue
        for buf_name in ('cos_cached', 'sin_cached'):
            if buf_name in re._buffers and re._buffers[buf_name] is not None:
                re._buffers[buf_name] = None
                count += 1
    return count


def process_file(path: Path, dry_run: bool = False, backup: bool = False) -> None:
    size_before = path.stat().st_size
    print(f"\n[{path.name}]  {size_before / 1e9:.3f} GB", end="", flush=True)

    obj = torch.load(path, map_location="cpu", weights_only=False)

    model = obj.get('model') if isinstance(obj, dict) else None
    if model is None:
        print("  — skipped (no 'model' key)")
        return

    n = strip_rope_cache(model)
    if n == 0:
        print("  — no rotary caches found, skipping")
        return

    print(f"  cleared {n} buffers", end="", flush=True)

    if dry_run:
        print("  [dry-run, not saved]")
        return

    if backup:
        bak = path.with_suffix('.pt.bak')
        path.rename(bak)
        print(f"  bak→{bak.name}", end="", flush=True)

    torch.save(obj, path)
    size_after = path.stat().st_size
    saved = (size_before - size_after) / 1e9
    print(f"  → {size_after / 1e9:.3f} GB  (saved {saved:.2f} GB)")


def collect_pts(targets, recursive: bool):
    paths = []
    for t in targets:
        p = Path(t)
        if p.is_file() and p.suffix == '.pt':
            paths.append(p)
        elif p.is_dir():
            pattern = '**/*.pt' if recursive else '*.pt'
            paths.extend(sorted(p.glob(pattern)))
        else:
            print(f"Warning: {t} not found, skipping")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Strip RoPE cache from SVDLLM checkpoints")
    parser.add_argument('targets', nargs='+', help='checkpoint file(s) or directory')
    parser.add_argument('--recursive', action='store_true', help='recurse into subdirectories')
    parser.add_argument('--dry-run', action='store_true', help='report sizes without modifying files')
    parser.add_argument('--backup', action='store_true', help='rename original to .pt.bak before overwriting')
    args = parser.parse_args()

    paths = collect_pts(args.targets, args.recursive)
    if not paths:
        print("No .pt files found.")
        sys.exit(0)

    print(f"Found {len(paths)} checkpoint(s). dry_run={args.dry_run} backup={args.backup}")
    for p in paths:
        process_file(p, dry_run=args.dry_run, backup=args.backup)
    print("\nDone.")


if __name__ == '__main__':
    main()
