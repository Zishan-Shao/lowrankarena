"""
Uniform Rank Allocation
Assign the same compression rank to all layers (uniform compression).
"""
import torch
import pickle as pk
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_model_path', type=str, required=True,
                        help='Path to local model directory')
    parser.add_argument('--svd_file', type=str, required=True,
                        help='Path to SVD results file (.pk)')
    parser.add_argument('--compression_ratio', type=float, required=True,
                        help='Fraction of factor parameters retained (e.g., 0.8 keeps 80%)')
    parser.add_argument('--output_file', type=str, default='rank_allocation_uniform.pk',
                        help='Output file for rank allocation')
    return parser.parse_args()


def rank_from_keep_ratio(out_features, in_features, keep_ratio):
    """Return r such that r*(out+in) <= keep_ratio*out*in.

    Swift-SVD originally used ``hidden_size * ratio / 2`` for every
    attention projection.  That is correct for square LLaMA-1 matrices but
    over-allocates the rectangular K/V projections used by GQA models such
    as Llama-3.1.  Computing the rank from each matrix shape preserves the
    benchmark's parameter-count keep-ratio contract.
    """
    if not 0.0 < float(keep_ratio) <= 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    out_features = int(out_features)
    in_features = int(in_features)
    rank = int(float(keep_ratio) * out_features * in_features / (out_features + in_features))
    return max(1, min(rank, out_features, in_features))


def projection_shapes(config):
    """Canonical (out_features, in_features) shapes for one decoder block."""
    hidden = int(config.hidden_size)
    model_type = getattr(config, "model_type", "").lower()
    if model_type == "opt":
        ffn = int(config.ffn_dim)
        return {
            "query": (hidden, hidden), "key": (hidden, hidden),
            "value": (hidden, hidden), "output": (hidden, hidden),
            "fc1": (ffn, hidden), "fc2": (hidden, ffn),
        }

    intermediate = int(config.intermediate_size)
    head_dim = int(getattr(config, "head_dim", hidden // int(config.num_attention_heads)))
    q_out = int(config.num_attention_heads) * head_dim
    kv_out = int(getattr(config, "num_key_value_heads", config.num_attention_heads)) * head_dim
    return {
        "query": (q_out, hidden), "key": (kv_out, hidden),
        "value": (kv_out, hidden), "output": (hidden, q_out),
        "gate": (intermediate, hidden), "up": (intermediate, hidden),
        "down": (hidden, intermediate),
    }


def compute_frobenius_loss_at_rank(svd_module, rank):
    """
    Compute Frobenius loss at a specified rank.
    
    Args:
        svd_module: IncrementalSVD object
        rank: Compression rank
    
    Returns:
        loss: Frobenius loss
    """
    S = svd_module.S  # shape: (D,)
    D = S.shape[0]
    
    if rank >= D:
        return 0.0
    
    # Loss(r) = sqrt(sum(S[r:]^2))
    # i.e., sqrt of the sum of squared discarded singular values
    loss = torch.sqrt(torch.sum(S[rank:] ** 2)).item()
    return loss


def uniform_allocation(svd_list, compression_ratio, config):
    """
    Uniformly allocate rank (all layers use the same rank).
    
    Args:
        svd_list: List of per-layer SVD results
        compression_ratio: Global compression ratio for Key (0-1)
        config: Model config object
    
    Returns:
        rank_allocation: Allocated rank per layer (key and value)
    
        Allocate uniform ranks for the 7 projection layers based on the ratio and formula.
    """

    n_layers = len(svd_list)
    
    # 1. Get dimensions from config (supports Llama, OPT, Phi-3, etc.)
    shapes = projection_shapes(config)
    target_ranks = {
        name: rank_from_keep_ratio(out_features, in_features, compression_ratio)
        for name, (out_features, in_features) in shapes.items()
    }
    
    print(f"\n{'='*60}")
    print(f"Allocation Summary (Uniform Allocation)")
    print(f"{'='*60}")
    for name, shape in shapes.items():
        print(f"{name:>8}: {shape[0]}x{shape[1]} -> rank {target_ranks[name]}")
    print(f"{'='*60}\n")
    
    # 3. Define mappings based on model architecture
    # Check model type (via config.model_type or architecture)
    model_type = getattr(config, "model_type", "").lower()
    
    # Check if model is OPT (OPT uses fc1/fc2)
    is_opt = "opt" in model_type
    
    if is_opt:
        # OPT model: q_proj, k_proj, v_proj, out_proj + fc1, fc2
        targets_config = target_ranks
        print(f"Detected OPT model architecture; using fc1/fc2 config\n")
    else:
        # LLaMA/Mistral/Phi models: q/k/v/o + gate, up, down
        targets_config = target_ranks
        print(f"Detected LLaMA/Mistral/Phi architecture; using gate/up/down config\n")

    rank_allocation = []
    
    # 4. Process each layer
    for i in tqdm(range(n_layers), desc="Computing Frobenius Loss"):
        layer_data = {'layer': i}
        
        for name, r in targets_config.items():
            # Create the key required by apply_full_svd_redemption: '{name}_rank'
            layer_data[f'{name}_rank'] = r
            
            # Compute and store the estimated Frobenius loss
            # Note: svd_list[i] must contain the 7 corresponding keys
            loss = compute_frobenius_loss_at_rank(svd_list[i][name], r)
            layer_data[f'{name}_loss'] = loss
            
        rank_allocation.append(layer_data)
        
    return rank_allocation


if __name__ == "__main__":
    args = parse_args()
    
    # Load SVD results
    print(f"Loading SVD results from: {args.svd_file}")
    svd_list = pk.load(open(args.svd_file, 'rb'))

    # 1. Load config only (no model weights)
    # args.local_model_path should point to a folder containing config.json
    config = AutoConfig.from_pretrained(args.local_model_path)
    
    # Run uniform rank allocation
    rank_allocation = uniform_allocation(
        svd_list, 
        args.compression_ratio, 
        config
    )
    
    # Save results
    print(f"Saving rank allocation to: {args.output_file}")
    pk.dump(rank_allocation, open(args.output_file, 'wb'))
    
    print("\n✓ Done!")
