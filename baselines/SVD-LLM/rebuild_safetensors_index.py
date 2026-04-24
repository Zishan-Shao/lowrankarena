"""
rebuild_safetensors_index.py
----------------------------
Rebuild a missing model.safetensors.index.json from existing shard files.

Usage:
    python rebuild_safetensors_index.py <checkpoint_dir>

Example:
    python rebuild_safetensors_index.py \
        /home/ww247/lowrankarena/hf_ckpts/LowRankArena/llama2_7b/BasisSharing/meta_llama_Llama_2_7b_hf_basis_sharing_0.8
"""

import json
import sys
from pathlib import Path

from safetensors import safe_open


def rebuild_index(model_dir: str | Path) -> Path:
    model_dir = Path(model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {model_dir}")

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        print(f"[info] index already exists: {index_path}")
        return index_path

    shard_files = sorted(model_dir.glob("model-*.safetensors"))
    if not shard_files:
        # Check if single model.safetensors exists — no index needed
        single = model_dir / "model.safetensors"
        if single.exists():
            print(f"[info] single-file checkpoint found — no index needed: {single}")
            return single
        raise FileNotFoundError(f"No safetensors shard files found in {model_dir}")

    print(f"[rebuild] found {len(shard_files)} shard(s) in {model_dir}")

    weight_map: dict[str, str] = {}
    total_size = 0

    for shard_file in shard_files:
        print(f"  scanning {shard_file.name} ...", end=" ", flush=True)
        with safe_open(str(shard_file), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                tensor = f.get_tensor(key)
                total_size += tensor.numel() * tensor.element_size()
                weight_map[key] = shard_file.name
        print(f"{len(keys)} tensors")

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }

    with open(index_path, "w") as fp:
        json.dump(index, fp, indent=2, sort_keys=True)

    print(f"\n[done] written: {index_path}")
    print(f"  {len(weight_map)} tensors across {len(shard_files)} shards")
    print(f"  total size: {total_size / 1e9:.2f} GB")
    return index_path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    rebuild_index(sys.argv[1])


if __name__ == "__main__":
    main()
