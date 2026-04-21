"""
convert_pt_to_hf_dir.py
-----------------------
Convert SVDLLM .pt checkpoints to HF-dir format loadable by eval_decoder.py.

Output directory structure:
    {stem}/
        model.pt           — model object only (for fast torch.load)
        tokenizer_config.json, tokenizer.model, ...
        config.json
        lowrank_config.json  — {"framework": "svdllm"} so eval_decoder adds sys.path

Usage:
    # Single file
    python convert_pt_to_hf_dir.py checkpoints/svdllm/llama2_7b/meta_llama_Llama_2_7b_hf_v2hetero_0.8.pt

    # Whole directory (all *.pt matching pattern)
    python convert_pt_to_hf_dir.py checkpoints/svdllm/llama2_7b/ --pattern "*v2hetero*"

    # Custom output root
    python convert_pt_to_hf_dir.py model.pt --output_dir /tmp/hf_dirs/
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch


def convert(pt_path: Path, output_root: Path | None = None) -> Path:
    stem = pt_path.stem
    out_dir = (output_root / stem) if output_root else (pt_path.parent / stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[convert] {pt_path.name}  ({pt_path.stat().st_size / 1e9:.2f} GB)")

    print("  loading ...", end=" ", flush=True)
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        tokenizer = obj.get("tokenizer")
    elif hasattr(obj, "forward"):
        model = obj
        tokenizer = None
    else:
        raise ValueError(f"Unrecognized checkpoint format in {pt_path}")
    print("done")

    # Save model.pt (model only, no tokenizer — keeps file smaller)
    model_pt = out_dir / "model.pt"
    print(f"  saving model.pt ...", end=" ", flush=True)
    torch.save(model, model_pt)
    print(f"done  ({model_pt.stat().st_size / 1e9:.2f} GB)")

    # Save config.json
    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg.save_pretrained(str(out_dir))
        print("  config.json saved")

    # Save lowrank_config.json so eval_decoder.py adds SVD-LLM to sys.path
    lowrank_cfg = {"framework": "svdllm"}
    with open(out_dir / "lowrank_config.json", "w") as f:
        json.dump(lowrank_cfg, f, indent=2)
    print("  lowrank_config.json saved")

    # Save tokenizer
    if tokenizer is not None:
        try:
            tokenizer.save_pretrained(str(out_dir))
            print("  tokenizer saved")
        except Exception as e:
            print(f"  [warn] tokenizer.save_pretrained failed: {e}")
            model_id = getattr(tokenizer, "name_or_path", None)
            if model_id:
                try:
                    from transformers import AutoTokenizer
                    AutoTokenizer.from_pretrained(model_id, trust_remote_code=True).save_pretrained(str(out_dir))
                    print(f"  tokenizer reloaded from {model_id} and saved")
                except Exception as e2:
                    print(f"  [warn] tokenizer reload failed: {e2}")
    else:
        print("  [warn] no tokenizer in checkpoint")

    del model, obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  → {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="+", help=".pt file(s) or directory")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--pattern", default="*.pt",
                        help="Glob pattern when target is a directory (default: *.pt)")
    args = parser.parse_args()

    pts: list[Path] = []
    for t in args.targets:
        p = Path(t)
        if p.is_file() and p.suffix == ".pt":
            pts.append(p)
        elif p.is_dir():
            pts.extend(sorted(p.glob(args.pattern)))
        else:
            print(f"Warning: {t} not found or not a .pt file, skipping")

    if not pts:
        print("No .pt files found.")
        sys.exit(0)

    print(f"Found {len(pts)} checkpoint(s).")
    output_root = Path(args.output_dir) if args.output_dir else None
    for pt in pts:
        try:
            convert(pt, output_root=output_root)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print("\nAll done.")


if __name__ == "__main__":
    main()
