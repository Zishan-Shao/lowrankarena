"""
convert_pt_to_safetensors.py
-----------------------------
Convert SVDLLM .pt checkpoints (pickled model objects) to a directory containing:
  - model.safetensors   — state dict tensors (memory-mappable, no pickle)
  - config.json         — model config
  - svd_metadata.json   — SVD layer architecture (ratio/ranks per layer, needed for rebuild)
  - tokenizer files     — copied from the tokenizer object

Usage:
    # Single file
    python convert_pt_to_safetensors.py checkpoints/svdllm/llama31_8b/meta_llama_Llama_3.1_8B_v2_0.8.pt

    # Directory (converts all *.pt files)
    python convert_pt_to_safetensors.py checkpoints/svdllm/llama31_8b/

    # Specify output root (default: same dir as input, subdirectory per checkpoint)
    python convert_pt_to_safetensors.py model.pt --output_dir /tmp/safetensors/

Loading the converted checkpoint:
    from convert_pt_to_safetensors import load_safetensors_checkpoint
    model, tokenizer = load_safetensors_checkpoint("path/to/output_dir")
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_svd_metadata(model) -> dict:
    """Walk model layers and record SVD rank info per layer.

    For each decoder layer, records the shape of q_u_proj weight so that
    the rank can be recovered without re-running whitening_hetero.
    """
    meta = {"layers": {}}
    layers = None
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            layers = inner.layers
        elif hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
            layers = inner.decoder.layers

    if layers is None:
        return meta

    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        entry = {}

        if attn is not None:
            for proj in ("q_u_proj", "k_u_proj", "v_u_proj", "o_u_proj"):
                w = getattr(attn, proj, None)
                if w is not None and hasattr(w, "weight"):
                    entry[f"attn.{proj}.shape"] = list(w.weight.shape)
            attn_type = type(attn).__name__
            entry["attn_type"] = attn_type

        if mlp is not None:
            for proj in ("gate_u_proj", "up_u_proj", "down_u_proj"):
                w = getattr(mlp, proj, None)
                if w is not None and hasattr(w, "weight"):
                    entry[f"mlp.{proj}.shape"] = list(w.weight.shape)
            mlp_type = type(mlp).__name__
            entry["mlp_type"] = mlp_type

        meta["layers"][str(i)] = entry

    meta["model_type"] = type(model).__name__
    cfg = getattr(model, "config", None)
    if cfg is not None:
        meta["model_name_or_path"] = getattr(cfg, "_name_or_path", None)
        meta["num_hidden_layers"] = getattr(cfg, "num_hidden_layers", None)

    return meta


def _state_dict_to_saveable(model, dtype=None) -> dict[str, torch.Tensor]:
    """Extract state dict, skipping None buffers and making tensors contiguous.

    Always clones each tensor so that basis-sharing checkpoints (where
    q_v_proj / k_v_proj / v_v_proj share the same storage) produce
    independent tensors that safetensors can serialize without error.

    If dtype is given (e.g. torch.bfloat16), floating-point tensors are cast
    to that dtype. Integer/bool tensors are left unchanged.
    """
    sd = {}
    for k, v in model.state_dict().items():
        if v is None:
            continue
        if not isinstance(v, torch.Tensor):
            continue
        # .clone() materialises shared-storage tensors (Basis Sharing V matrices)
        # and also ensures contiguity, so no separate .contiguous() needed.
        t = v.detach().clone().cpu()
        if dtype is not None and t.is_floating_point():
            t = t.to(dtype)
        sd[k] = t
    return sd


def _save_tokenizer(tokenizer, out_dir: Path) -> None:
    """Save tokenizer files to out_dir."""
    try:
        tokenizer.save_pretrained(str(out_dir))
    except Exception as e:
        print(f"  [warn] tokenizer.save_pretrained failed: {e}")
        # Fallback: try to reload from model_id and save that
        model_id = getattr(tokenizer, "name_or_path", None)
        if model_id:
            try:
                from transformers import AutoTokenizer
                tok2 = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                tok2.save_pretrained(str(out_dir))
                print(f"  [info] tokenizer reloaded from {model_id} and saved")
            except Exception as e2:
                print(f"  [warn] tokenizer reload also failed: {e2}")


def convert_checkpoint(pt_path: Path, output_root: Path | None = None,
                       dtype: torch.dtype | None = None) -> Path:
    """Convert a single .pt checkpoint to a safetensors directory.

    Returns the output directory path.
    """
    from safetensors.torch import save_file

    print(f"\n[convert] {pt_path.name}  ({pt_path.stat().st_size / 1e9:.2f} GB)")

    # Determine output directory
    stem = pt_path.stem  # e.g. meta_llama_Llama_3.1_8B_v2_0.8
    if output_root is None:
        out_dir = pt_path.parent / stem
    else:
        out_dir = output_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint (CPU to avoid OOM)
    print("  loading checkpoint ...", end=" ", flush=True)
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        tokenizer = obj.get("tokenizer")
    elif hasattr(obj, "forward"):
        model = obj
        tokenizer = None
    else:
        raise ValueError(f"Unrecognized checkpoint format: {pt_path}")
    print("done")

    # Collect SVD architecture metadata
    print("  collecting SVD metadata ...", end=" ", flush=True)
    svd_meta = _collect_svd_metadata(model)
    with open(out_dir / "svd_metadata.json", "w") as f:
        json.dump(svd_meta, f, indent=2)
    print(f"done  ({len(svd_meta.get('layers', {}))} layers)")

    # Save config
    cfg = getattr(model, "config", None)
    if cfg is not None:
        try:
            cfg.save_pretrained(str(out_dir))
        except TypeError:
            # Newer transformers stores torch_dtype as a dtype object, not a string.
            # Manually serialize after converting dtype objects → strings.
            import json as _json
            cfg_dict = cfg.to_dict()
            for _k, _v in list(cfg_dict.items()):
                if isinstance(_v, torch.dtype):
                    cfg_dict[_k] = str(_v).replace("torch.", "")
            with open(out_dir / "config.json", "w") as _f:
                _json.dump(cfg_dict, _f, indent=2)
        print("  config saved")
    else:
        print("  [warn] no model.config found")

    # Build state dict and save as safetensors
    dtype_label = str(dtype).replace("torch.", "") if dtype is not None else "unchanged"
    print(f"  building state dict (dtype={dtype_label}) ...", end=" ", flush=True)
    tensors = _state_dict_to_saveable(model, dtype=dtype)
    n_params = sum(t.numel() for t in tensors.values())
    print(f"{len(tensors)} tensors, {n_params / 1e9:.2f}B params")

    safetensors_path = out_dir / "model.safetensors"
    print(f"  saving safetensors → {safetensors_path.name} ...", end=" ", flush=True)
    save_file(tensors, str(safetensors_path))
    size_gb = safetensors_path.stat().st_size / 1e9
    print(f"done  ({size_gb:.2f} GB)")

    # Free memory
    del tensors, model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save tokenizer
    if tokenizer is not None:
        print("  saving tokenizer ...", end=" ", flush=True)
        _save_tokenizer(tokenizer, out_dir)
        print("done")
    else:
        print("  [warn] no tokenizer in checkpoint")

    print(f"  → {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Loading helper
# ---------------------------------------------------------------------------

def load_safetensors_checkpoint(directory: str | Path):
    """Load a converted safetensors checkpoint.

    This rebuilds the SVD architecture from svd_metadata.json and then
    loads the state dict from model.safetensors.

    Returns (model, tokenizer).
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoTokenizer

    directory = Path(directory)

    # Load metadata
    meta_path = directory / "svd_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"svd_metadata.json not found in {directory}")
    with open(meta_path) as f:
        svd_meta = json.load(f)

    model_name_or_path = svd_meta.get("model_name_or_path")
    if model_name_or_path is None:
        raise ValueError("model_name_or_path not found in svd_metadata.json")

    # Load config (from saved config.json, falls back to HF hub)
    config = AutoConfig.from_pretrained(str(directory), trust_remote_code=True)

    # Infer per-layer ranks from metadata and reconstruct the SVD architecture
    print(f"Rebuilding SVD architecture for {model_name_or_path} ...")

    # Import the whitening function to rebuild the model skeleton
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    from SVDLLM_v2 import whitening_hetero
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model_name = model_name_or_path.split("/")[-1].lower()

    # Infer ratio from first layer's q_u_proj rank
    layers_meta = svd_meta.get("layers", {})
    ratio = None
    for layer_info in layers_meta.values():
        shape = layer_info.get("attn.q_u_proj.shape")
        if shape is not None:
            # q_u_proj: [out_dim, rank]; original out_dim = num_heads * head_dim
            rank = shape[1]
            max_rank = min(config.hidden_size, config.hidden_size)  # square for q_proj
            # For q_proj: [num_kv_heads*head_dim, rank] or [hidden_size, rank]
            # Use the rank directly divided by max(shape) as keep ratio
            max_rank = max(shape)
            ratio = rank / max_rank
            break
    if ratio is None:
        raise ValueError("Could not infer ratio from svd_metadata.json")
    print(f"  inferred keep ratio ≈ {ratio:.3f}")

    # Dummy profiling_mat (identity) for architecture reconstruction only
    # (weights will be overwritten by load_file below)
    num_layers = config.num_hidden_layers
    inner = model.model if hasattr(model, "model") else model
    layers = getattr(inner, "layers", getattr(getattr(inner, "decoder", None), "layers", None))

    dummy_prof = {}
    for i in range(num_layers):
        layer = layers[i]
        from utils.model_utils import find_layers
        subset = find_layers(layer)
        dummy_prof[i] = {n: torch.eye(subset[n].weight.shape[1]) for n in subset}

    whitening_hetero(
        model_name=model_name,
        model=model,
        profiling_mat=dummy_prof,
        ratio=ratio,
        dev=torch.device("cpu"),
    )

    # Load state dict from safetensors
    print("Loading state dict from safetensors ...")
    sd = load_file(str(directory / "model.safetensors"), device="cpu")
    model.load_state_dict(sd, strict=True)

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(directory), trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    return model, tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert SVDLLM .pt checkpoints to safetensors")
    parser.add_argument("targets", nargs="+", help=".pt file(s) or directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Root output directory (default: sibling directory of each .pt file)")
    parser.add_argument("--dry_run", action="store_true",
                        help="List files to convert without converting")
    parser.add_argument("--dtype", type=str, default=None,
                        help="Cast float tensors to this dtype (e.g. bf16, fp16, fp32). "
                             "Integer/bool tensors are always left unchanged.")
    args = parser.parse_args()

    try:
        from safetensors.torch import save_file  # noqa: F401
    except ImportError:
        print("ERROR: safetensors not installed. Run: pip install safetensors")
        sys.exit(1)

    # Collect .pt files
    pts: list[Path] = []
    for t in args.targets:
        p = Path(t)
        if p.is_file() and p.suffix == ".pt":
            pts.append(p)
        elif p.is_dir():
            pts.extend(sorted(p.glob("*.pt")))
        else:
            print(f"Warning: {t} not found or not a .pt file, skipping")

    if not pts:
        print("No .pt files found.")
        sys.exit(0)

    print(f"Found {len(pts)} checkpoint(s).")
    if args.dry_run:
        for p in pts:
            print(f"  {p}")
        return

    _dtype_map = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                  "fp16": torch.float16, "float16": torch.float16,
                  "fp32": torch.float32, "float32": torch.float32}
    cast_dtype = None
    if args.dtype is not None:
        cast_dtype = _dtype_map.get(args.dtype.lower())
        if cast_dtype is None:
            print(f"ERROR: unknown dtype '{args.dtype}'. Choose from: bf16, fp16, fp32")
            sys.exit(1)
        print(f"Will cast float tensors → {cast_dtype}")

    output_root = Path(args.output_dir) if args.output_dir else None
    for pt in pts:
        try:
            convert_checkpoint(pt, output_root=output_root, dtype=cast_dtype)
        except Exception as e:
            print(f"  ERROR converting {pt.name}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll done.")


if __name__ == "__main__":
    main()
