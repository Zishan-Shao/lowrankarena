#!/usr/bin/env python3
"""Quick test of ModernBERT v1 implementation."""

import sys
import torch
from profile_svdllm_v1 import (
    ModernBertSVDBlock,
    calibrate_covariances,
    _data_aware_low_rank,
    _data_aware_per_head,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

def test_architecture_access():
    """Test 1: Verify architecture access."""
    print("="*80)
    print("TEST 1: Architecture Access")
    print("="*80)

    model = AutoModelForSequenceClassification.from_pretrained(
        "answerdotai/ModernBERT-base",
        num_labels=2,
        trust_remote_code=True
    )

    # Verify encoder path
    assert hasattr(model, 'model'), "model.model not found!"
    assert hasattr(model.model, 'layers'), "model.model.layers not found!"
    print(f"✅ Encoder path verified: model.model.layers ({len(model.model.layers)} layers)")

    # Verify layer structure
    layer = model.model.layers[0]
    assert hasattr(layer, 'attn'), "layer.attn not found!"
    assert hasattr(layer.attn, 'Wqkv'), "layer.attn.Wqkv not found!"
    assert hasattr(layer, 'mlp'), "layer.mlp not found!"
    assert hasattr(layer.mlp, 'Wi'), "layer.mlp.Wi not found!"
    assert hasattr(layer.mlp, 'Wo'), "layer.mlp.Wo not found!"
    print(f"✅ Layer structure verified")

    # Verify shapes
    Wqkv_shape = layer.attn.Wqkv.weight.shape
    Wi_shape = layer.mlp.Wi.weight.shape
    Wo_shape = layer.mlp.Wo.weight.shape
    print(f"  Wqkv shape: {Wqkv_shape} (expected [2304, 768])")
    print(f"  Wi shape: {Wi_shape} (expected [2304, 768])")
    print(f"  Wo shape: {Wo_shape} (expected [768, 1152])")

    assert Wqkv_shape == torch.Size([2304, 768]), f"Unexpected Wqkv shape: {Wqkv_shape}"
    assert Wi_shape == torch.Size([2304, 768]), f"Unexpected Wi shape: {Wi_shape}"
    assert Wo_shape == torch.Size([768, 1152]), f"Unexpected Wo shape: {Wo_shape}"
    print(f"✅ All shapes correct!")

    del model
    torch.cuda.empty_cache()
    print()

def test_svd_factorization():
    """Test 2: SVD factorization on small tensors."""
    print("="*80)
    print("TEST 2: SVD Factorization")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test weight and covariance
    W = torch.randn(768, 768, device=device)
    C = torch.randn(768, 768, device=device)
    C = C @ C.t() + torch.eye(768, device=device) * 0.1  # make positive definite

    # Test factorization
    U, V = _data_aware_low_rank(W, rank=128, cov_in=C)
    print(f"  W shape: {W.shape}")
    print(f"  U shape: {U.shape} (expected [768, 128])")
    print(f"  V shape: {V.shape} (expected [128, 768])")

    assert U.shape == torch.Size([768, 128]), f"Unexpected U shape: {U.shape}"
    assert V.shape == torch.Size([128, 768]), f"Unexpected V shape: {V.shape}"

    # Check reconstruction error
    W_approx = U @ V
    rel_error = torch.norm(W - W_approx) / torch.norm(W)
    print(f"  Reconstruction relative error: {rel_error:.6f}")
    print(f"✅ SVD factorization works!")

    del W, C, U, V, W_approx
    torch.cuda.empty_cache()
    print()

def test_calibration():
    """Test 3: Calibration on small dataset."""
    print("="*80)
    print("TEST 3: Calibration")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "answerdotai/ModernBERT-base",
        num_labels=2,
        trust_remote_code=True
    ).to(device)

    # Load small dataset
    raw = load_dataset("glue", "sst2", split="validation[:8]")  # Just 8 samples
    tokz = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", trust_remote_code=True)

    def tokenize_fn(batch):
        return tokz(batch["sentence"], padding="max_length", truncation=True, max_length=128)

    ds = raw.map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"])
    ds.set_format("torch")
    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda b: {
            "input_ids":      torch.stack([x["input_ids"]      for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            "labels":         torch.tensor([x["label"]         for x in b]),
        },
    )

    print(f"  Running calibration with {len(ds)} samples...")
    covs = calibrate_covariances(model, loader, device, max_batches=2)

    # Check shapes
    print(f"  cov_attn_in: {len(covs['cov_attn_in'])} tensors, shape {covs['cov_attn_in'][0].shape}")
    print(f"  cov_attn_out: {len(covs['cov_attn_out'])} tensors, shape {covs['cov_attn_out'][0].shape}")
    print(f"  cov_ffn_in: {len(covs['cov_ffn_in'])} tensors, shape {covs['cov_ffn_in'][0].shape}")
    print(f"  cov_ffn_out: {len(covs['cov_ffn_out'])} tensors, shape {covs['cov_ffn_out'][0].shape}")

    assert len(covs['cov_attn_in']) == 22, f"Expected 22 layers, got {len(covs['cov_attn_in'])}"
    assert covs['cov_attn_in'][0].shape == torch.Size([768, 768])
    assert covs['cov_ffn_out'][0].shape == torch.Size([1152, 1152])

    print(f"✅ Calibration works!")

    del model, covs
    torch.cuda.empty_cache()
    print()

def test_svd_block():
    """Test 4: ModernBertSVDBlock creation and forward pass."""
    print("="*80)
    print("TEST 4: ModernBertSVDBlock")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "answerdotai/ModernBERT-base",
        num_labels=2,
        trust_remote_code=True
    ).to(device)

    # Create dummy covariances
    dm, d_ff = 768, 1152
    cov_attn_in = torch.eye(dm, device=device)
    cov_attn_out = torch.eye(dm, device=device)
    cov_ffn_in = torch.eye(dm, device=device)
    cov_ffn_out = torch.eye(d_ff, device=device)

    # Create SVD block
    layer = model.model.layers[0]
    print(f"  Creating SVD block for layer 0...")
    svd_block = ModernBertSVDBlock(
        hf_layer=layer,
        rank_attn=64,
        rank_ff=64,
        cov_attn_in=cov_attn_in,
        cov_attn_out=cov_attn_out,
        cov_ffn_in=cov_ffn_in,
        cov_ffn_out=cov_ffn_out,
        rank_wo=64,
        num_heads=12,
    ).to(device)

    print(f"✅ SVD block created!")

    # Test forward pass
    print(f"  Testing forward pass...")
    x = torch.randn(2, 16, 768, device=device)  # [B=2, M=16, dm=768]
    mask = torch.ones(2, 16, device=device, dtype=torch.bool)

    try:
        out = svd_block(x, mask)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        assert out.shape == x.shape, f"Output shape mismatch: {out.shape} vs {x.shape}"
        print(f"✅ Forward pass works!")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        raise

    del model, svd_block, x, out
    torch.cuda.empty_cache()
    print()

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ModernBERT v1 Quick Test Suite")
    print("="*80 + "\n")

    try:
        test_architecture_access()
        test_svd_factorization()
        test_calibration()
        test_svd_block()

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80 + "\n")

    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80 + "\n")
        raise
