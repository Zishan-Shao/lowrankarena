"""Convert a .pt checkpoint (saved as {"model": ..., "tokenizer": ...}) to HuggingFace format.

With --vllm_ready, also prepares the output for serving with vLLM by:
  - Setting architectures to LlamaLowRankForCausalLM
  - Adding low_rank_config with layer ranks extracted from safetensors
  - Copying modeling_llama_lowrank.py into the output directory
"""

import argparse
import glob
import json
import os
import shutil

import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    """Drop-in replacement for nn.Linear that factors weight as U @ V."""
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.v_proj = nn.Linear(in_features, rank, bias=False)
        self.u_proj = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.u_proj(self.v_proj(x))


def prepare_vllm_ready(output_dir: str, modeling_src: str = None):
    """Add vLLM-compatible config and modeling files to an HF checkpoint dir."""
    from safetensors import safe_open

    # --- Extract low_rank_config from safetensors ---
    shards = sorted(glob.glob(os.path.join(output_dir, "model-*.safetensors")))
    ranks = {}
    for shard in shards:
        f = safe_open(shard, framework="pt")
        for key in f.keys():
            if ".v_proj.weight" in key:
                if "self_attn.v_proj" in key:
                    if "self_attn.v_proj.weight" in key:
                        continue
                    else:
                        parts = key.replace("self_attn.v_proj.v_proj.weight", "self_attn.v_proj")
                        ranks[parts] = f.get_tensor(key).shape[0]
                else:
                    parts = key.replace(".v_proj.weight", "")
                    ranks[parts] = f.get_tensor(key).shape[0]

    # --- Update config.json ---
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    if ranks:
        config["architectures"] = ["LlamaLowRankForCausalLM"]
        config["auto_map"] = {
            "AutoModelForCausalLM": "modeling_llama_lowrank.LlamaLowRankForCausalLM"
        }
        config["low_rank_config"] = {"layers": ranks}
    # If no low-rank keys found, leave as standard LlamaForCausalLM

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # --- Copy modeling files ---
    if modeling_src is None:
        # Search common locations
        candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama-lowrank-fused"),
            os.path.dirname(os.path.abspath(__file__)),
        ]
        for cand in candidates:
            src = os.path.join(cand, "modeling_llama_lowrank.py")
            if os.path.isfile(src):
                modeling_src = cand
                break

    if modeling_src:
        for fname in ["modeling_llama_lowrank.py", "modeling_llama_lowrank_nofusion.py"]:
            src = os.path.join(modeling_src, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(output_dir, fname))
    else:
        print("  WARNING: Could not find modeling_llama_lowrank.py in any search path.")
        print("  The model will NOT be servable with vLLM until you copy the file manually.")
        print("  Use --modeling_src to specify the directory containing the file.")

    print(f"  vLLM ready: {len(ranks)} low-rank layers, "
          f"{len(shards)} shards, "
          f"modeling file: {'YES' if os.path.isfile(os.path.join(output_dir, 'modeling_llama_lowrank.py')) else 'NO'}")


def main():
    ap = argparse.ArgumentParser(description="Convert .pt checkpoint to HuggingFace save_pretrained format.")
    ap.add_argument("--ckpt_path", type=str, required=True, help="Path to the .pt checkpoint.")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="Output directory. Defaults to <ckpt_dir>/<ckpt_name>_hf/")
    ap.add_argument("--vllm_ready", action="store_true",
                    help="Also prepare for vLLM serving: add low_rank_config, architectures, modeling files.")
    ap.add_argument("--modeling_src", type=str, default=None,
                    help="Directory containing modeling_llama_lowrank.py. Auto-detected if not set.")
    args = ap.parse_args()

    if not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    # Resolve output dir
    if args.output_dir is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt_path))
        ckpt_name = os.path.splitext(os.path.basename(args.ckpt_path))[0]
        output_dir = os.path.join(ckpt_dir, f"{ckpt_name}_hf")
    else:
        output_dir = args.output_dir

    print(f"Loading checkpoint: {args.ckpt_path}")
    try:
        saved = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        saved = torch.load(args.ckpt_path, map_location="cpu")

    model = saved["model"]
    tokenizer = saved["tokenizer"]

    print(f"Saving HuggingFace model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.half().save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if args.vllm_ready:
        prepare_vllm_ready(output_dir, modeling_src=args.modeling_src)

    print("Done.")


if __name__ == "__main__":
    main()
