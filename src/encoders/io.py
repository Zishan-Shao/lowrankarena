#!/usr/bin/env python3
"""
Utility to load compressed SVD models for fine-tuning.

The key insight: compressed models are saved with SVD blocks already in place,
but AutoModelForSequenceClassification.from_pretrained() doesn't recognize
the custom structure. We need to manually reconstruct the SVD blocks.
"""

import json
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

# Add repo root to path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import copy as _copy
from src.encoders.blocks_misc import (
    BertLayerShim, ModernBertLayerShim, NaiveSVDBlock, _apply_rotary,
)


class _LowRankLinear(nn.Module):
    """Drop-in for nn.Linear: W ≈ A @ Bt. Matches adasvd_origin compress_adasvd_naive."""
    def __init__(self, A: torch.Tensor, Bt: torch.Tensor,
                 bias: "torch.Tensor | None" = None):
        super().__init__()
        self.A    = nn.Parameter(A,  requires_grad=False)
        self.Bt   = nn.Parameter(Bt, requires_grad=False)
        self.bias = nn.Parameter(bias.detach().clone(),
                                 requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.Bt.t() @ self.A.t()
        if self.bias is not None:
            out = out + self.bias
        return out


def load_compressed_model(
    checkpoint_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Tuple[nn.Module, AutoTokenizer, dict]:
    """
    Load a compressed SVD model from disk.

    Args:
        checkpoint_path: Path to saved compressed model directory
        device: Device to load model on
        dtype: Data type for model weights

    Returns:
        (model, tokenizer, compression_info)
    """
    checkpoint_path = Path(checkpoint_path).resolve()  # Use absolute path

    # Load compression metadata (if exists, otherwise assume dense)
    info_file = checkpoint_path / "compression_info.json"
    if info_file.exists():
        with open(info_file) as f:
            comp_info = json.load(f)
        print(f"[load] Loading compressed model: {checkpoint_path.name}")
        print(f"[info] Method: {comp_info['method']}, Rank: {comp_info.get('rank', 'N/A')}")
        print(f"[info] Backend: {comp_info['backend']}")
        print(f"[info] Original accuracy: {comp_info['accuracy_before_finetune']:.4f}")
    else:
        # Dense model or model without compression info
        # Try to read config to get model_id
        print(f"[load] Loading model: {checkpoint_path.name}")

        config_file = checkpoint_path / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config_data = json.load(f)
            model_id = config_data.get("_name_or_path", "bert-base-uncased")
        else:
            model_id = "bert-base-uncased"

        # Peek at state_dict keys to detect SVD checkpoints saved without compression_info.json
        # (can happen when an older version of the code was used, or if the save was interrupted)
        state_dict_path_peek = checkpoint_path / "pytorch_model.bin"
        detected_method = "dense"
        if state_dict_path_peek.exists():
            sd_peek = torch.load(state_dict_path_peek, map_location="cpu")
            if any((".block.Uq" in k or ".block.Pq" in k) for k in sd_peek.keys()):
                detected_method = "svd"
                print(f"[info] Detected SVD keys in checkpoint despite missing compression_info.json")
                print(f"[info] Treating checkpoint as compressed model (method=svd, backend=naive)")
            else:
                print(f"[info] No compression_info.json found, assuming dense/finetuned model")

        comp_info = {
            "method": detected_method,
            "backend": "naive",
            "rank": None,
            "model_id": model_id,
            "accuracy_before_finetune": 0.0,
        }

    # Load tokenizer from original model_id (not from checkpoint path)
    # This avoids HuggingFace validation errors with local paths
    model_id = comp_info.get('model_id', 'bert-base-uncased')
    print(f"[load] Loading tokenizer from: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # If dense model, load from original model_id and then load state dict
    if comp_info['method'] == 'dense':
        print(f"[load] Loading dense model from checkpoint")

        # Read config directly from file (avoid path validation issues)
        config_file = checkpoint_path / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config_dict = json.load(f)
            # Update model_id if available
            if '_name_or_path' in config_dict:
                model_id = config_dict['_name_or_path']
                comp_info['model_id'] = model_id

        # Create model from original model_id
        print(f"[load] Creating model from: {model_id}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        # Load state dict from checkpoint
        state_dict_path = checkpoint_path / "pytorch_model.bin"
        if state_dict_path.exists():
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"[load] Loaded state dict from checkpoint")
        else:
            print(f"[warn] No state dict found, using original model weights")

        model = model.to(device).to(dtype)
        print(f"[load] Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
        return model, tokenizer, comp_info

    # For compressed models, rebuild SVD blocks
    # Load the state dict directly (support both safetensors and legacy .bin)
    safetensors_path = checkpoint_path / "model.safetensors"
    bin_path = checkpoint_path / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file as safetensors_load
        state_dict = safetensors_load(str(safetensors_path), device="cpu")
    else:
        state_dict = torch.load(bin_path, map_location="cpu")

    # Detect architecture from state_dict keys
    if any("roberta" in k for k in state_dict.keys()):
        arch = "roberta"
        encoder_prefix = "roberta.encoder.layer"
    elif any("model.layers" in k for k in state_dict.keys()):
        arch = "modernbert"
        encoder_prefix = "model.layers"
    else:
        arch = "bert"
        encoder_prefix = "bert.encoder.layer"

    print(f"[load] Detected architecture: {arch}")

    # Read config directly from file (avoid path validation issues)
    config_file = checkpoint_path / "config.json"
    with open(config_file) as f:
        config_dict = json.load(f)

    # Update model_id if available in config
    if '_name_or_path' in config_dict:
        model_id_from_config = config_dict['_name_or_path']
        if comp_info.get('model_id') == 'bert-base-uncased':
            comp_info['model_id'] = model_id_from_config
            model_id = model_id_from_config

    # Create config object from model_id
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Update config with values from saved config
    for key, value in config_dict.items():
        if key not in ['_name_or_path', 'transformers_version']:
            setattr(config, key, value)

    # Create a fresh base model (this will have standard layers)
    print(f"[load] Creating base model from: {model_id}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
    )

    # ── AdaSVD-origin branch: _LowRankLinear format (A / Bt keys) ────────────
    # adasvd_origin replaces every nn.Linear with _LowRankLinear(A, Bt).
    # Checkpoint keys look like "bert.encoder.layer.0.attention.self.query.A".
    # No ".block." prefix → the SVD-block reconstruction below would skip all layers.
    has_lrl = any(k.endswith(".A") for k in state_dict.keys())
    if has_lrl:
        print("[load] Detected _LowRankLinear format (adasvd_origin) — restoring A/Bt layers")
        replaced = 0
        for name, module in list(base_model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            A_key  = f"{name}.A"
            Bt_key = f"{name}.Bt"
            if A_key not in state_dict or Bt_key not in state_dict:
                continue
            bias_val = state_dict.get(f"{name}.bias")
            lrl = _LowRankLinear(state_dict[A_key], state_dict[Bt_key], bias_val)
            *parts, last = name.split(".")
            parent = base_model
            for part in parts:
                parent = getattr(parent, part)
            setattr(parent, last, lrl)
            replaced += 1
        print(f"[load] Replaced {replaced} Linear ops with _LowRankLinear")
        # Load all remaining params (embeddings, layernorms, classifier, pooler)
        non_lrl = {k: v for k, v in state_dict.items()
                   if not k.endswith(".A") and not k.endswith(".Bt")}
        missing, unexpected = base_model.load_state_dict(non_lrl, strict=False)
        # A/Bt are already set in _LowRankLinear constructor — expected to be "missing" here.
        real_missing = [k for k in missing if not k.endswith(".A") and not k.endswith(".Bt")]
        if real_missing:
            print(f"[warn] Truly missing keys after adasvd reload: {len(real_missing)}")
            for k in real_missing[:10]:
                print(f"  - {k}")
        # Enable gradients on A/Bt so the factorised encoder can be fine-tuned.
        # (requires_grad=False was set for inference-only benchmarking in compress_adasvd_naive)
        for m in base_model.modules():
            if isinstance(m, _LowRankLinear):
                m.A.requires_grad_(True)
                m.Bt.requires_grad_(True)
        base_model = base_model.to(device=device, dtype=dtype).eval()
        trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in base_model.parameters())
        print(f"[load] adasvd model loaded: {total/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)")
        return base_model, tokenizer, comp_info
    # ── END adasvd-origin branch ──────────────────────────────────────────────

    # Detect number of layers from state_dict
    # Works for all architectures:
    #   BERT/RoBERTa: "bert.encoder.layer.N.block.X"  → prefix = "bert.encoder.layer"
    #   ModernBERT:   "model.layers.N.block.X"         → prefix = "model.layers"
    prefix_with_dot = encoder_prefix + "."
    prefix_len = len(prefix_with_dot)
    layer_indices = set()
    for key in state_dict.keys():
        if key.startswith(prefix_with_dot):
            rest = key[prefix_len:]          # "N.block.X"
            idx_str = rest.split(".")[0]
            try:
                layer_indices.add(int(idx_str))
            except ValueError:
                pass

    num_layers = max(layer_indices) + 1 if layer_indices else 12
    print(f"[load] Found {num_layers} encoder layers")

    # Get encoder layers reference
    if arch == "roberta":
        encoder_layers = base_model.roberta.encoder.layer
    elif arch == "modernbert":
        encoder_layers = base_model.model.layers
    else:
        encoder_layers = base_model.bert.encoder.layer

    # Replace each layer with a minimal SVD block shell and load parameters
    print("[load] Creating SVD block structure and loading parameters...")
    loaded_params = 0

    if arch == "modernbert":
        for i in range(num_layers):
            layer_prefix = f"{encoder_prefix}.{i}.block."
            has_svd_params = (
                f"{layer_prefix}Pq" in state_dict or
                f"{layer_prefix}Uq" in state_dict
            )
            if not has_svd_params:
                print(f"[warn] Layer {i} missing SVD parameters, skipping")
                continue

            blk = MinimalModernBertSVDBlock()
            hf_layer = encoder_layers[i]  # original ModernBertEncoderLayer

            # Metadata
            blk.num_heads = base_model.config.num_attention_heads
            blk.head_dim = base_model.config.hidden_size // blk.num_heads
            _rope = getattr(hf_layer.attn, "rotary_emb", None)
            blk.rotary_emb = _rope if isinstance(_rope, nn.Module) else base_model.model.rotary_emb  # shared reference
            blk.gelu_approximate = "none"

            # ffn_is_geglu: V1.shape[1] == 2 * U2.shape[0]
            # V1: [R, Wi.out_features]  U2: [Wo.in_features, R]
            # GeGLU: Wi.out_features = 2 * intermediate_size, Wo.in_features = intermediate_size
            v1_key = f"{layer_prefix}V1"
            u2_key = f"{layer_prefix}U2"
            if v1_key in state_dict and u2_key in state_dict:
                blk.ffn_is_geglu = (state_dict[v1_key].shape[1] == 2 * state_dict[u2_key].shape[0])
            else:
                blk.ffn_is_geglu = False

            # Required SVD params
            for pname in ("Pq", "Vq", "Pk", "Vk", "Pv", "Vv", "U1", "V1", "U2", "V2"):
                key = f"{layer_prefix}{pname}"
                if key in state_dict:
                    setattr(blk, pname, nn.Parameter(state_dict[key]))
                    loaded_params += 1
                else:
                    print(f"[warn] Layer {i} missing parameter: {pname}")

            # Optional bias params
            for pname in ("bq", "bk", "bv", "b1", "b2"):
                key = f"{layer_prefix}{pname}"
                if key in state_dict:
                    setattr(blk, pname, nn.Parameter(state_dict[key]))
                    loaded_params += 1

            # attn_norm
            w_key = f"{layer_prefix}attn_norm.weight"
            b_key = f"{layer_prefix}attn_norm.bias"
            if w_key in state_dict:
                blk.attn_norm = nn.LayerNorm(state_dict[w_key].shape[0])
                blk.attn_norm.weight = nn.Parameter(state_dict[w_key])
                if b_key in state_dict:
                    blk.attn_norm.bias = nn.Parameter(state_dict[b_key])
                loaded_params += (2 if b_key in state_dict else 1)
            else:
                blk.attn_norm = _copy.deepcopy(hf_layer.attn_norm)

            # mlp_norm
            w_key = f"{layer_prefix}mlp_norm.weight"
            b_key = f"{layer_prefix}mlp_norm.bias"
            if w_key in state_dict:
                blk.mlp_norm = nn.LayerNorm(state_dict[w_key].shape[0])
                blk.mlp_norm.weight = nn.Parameter(state_dict[w_key])
                if b_key in state_dict:
                    blk.mlp_norm.bias = nn.Parameter(state_dict[b_key])
                loaded_params += (2 if b_key in state_dict else 1)
            else:
                blk.mlp_norm = _copy.deepcopy(hf_layer.mlp_norm)

            # Wo_attn (SVD factorized: Uo/Vo/bo_attn, same as BERT Wo)
            for pname in ("Uo", "Vo"):
                key = f"{layer_prefix}{pname}"
                if key in state_dict:
                    setattr(blk, pname, nn.Parameter(state_dict[key]))
                    loaded_params += 1
                else:
                    print(f"[warn] Layer {i} missing parameter: {pname}")
            bo_key = f"{layer_prefix}bo_attn"
            if bo_key in state_dict:
                blk.bo_attn = nn.Parameter(state_dict[bo_key])
                loaded_params += 1

            encoder_layers[i] = ModernBertLayerShim(blk, attention_type=getattr(hf_layer, "attention_type", "global"))

    else:
        for i in range(num_layers):
            layer_prefix = f"{encoder_prefix}.{i}.block."

            # Check if this layer has SVD parameters (support both Pq and Uq formats)
            has_svd_params = (
                f"{layer_prefix}Pq" in state_dict or
                f"{layer_prefix}Uq" in state_dict
            )
            if not has_svd_params:
                print(f"[warn] Layer {i} missing SVD parameters, skipping")
                continue

            # Create minimal SVD block
            svd_block = MinimalSVDBlock()
            svd_block.num_heads = base_model.config.num_attention_heads  # Store for 2D path

            # Manually load all SVD parameters from state_dict
            # Support both formats:
            # 1. Full-matrix format: Uq/Vq, Uk/Vk, Uv/Vv (with bq_full, bk_full, bv_full)
            # 2. Per-head format: Pq/Vq, Pk/Vk, Pv/Vv (with bq, bk, bv)
            param_names = [
                # Q projection (try both Pq and Uq)
                ("Pq", ["Pq", "Uq"]), ("Vq", ["Vq"]), ("bq", ["bq", "bq_full"]),
                # K projection (try both Pk and Uk)
                ("Pk", ["Pk", "Uk"]), ("Vk", ["Vk"]), ("bk", ["bk", "bk_full"]),
                # V projection (try both Pv and Uv)
                ("Pv", ["Pv", "Uv"]), ("Vv", ["Vv"]), ("bv", ["bv", "bv_full"]),
                # Output projection
                ("Uo", ["Uo"]), ("Vo", ["Vo"]), ("bo_attn", ["bo_attn"]),
                # FFN
                ("U1", ["U1"]), ("V1", ["V1"]), ("b1", ["b1"]),
                ("U2", ["U2"]), ("V2", ["V2"]), ("b2", ["b2"])
            ]

            for target_name, candidate_names in param_names:
                loaded = False
                for candidate_name in candidate_names:
                    full_key = f"{layer_prefix}{candidate_name}"
                    if full_key in state_dict:
                        setattr(svd_block, target_name, nn.Parameter(state_dict[full_key]))
                        loaded_params += 1
                        loaded = True
                        break
                # Only warn if it's a critical parameter (not bias which might be missing)
                if not loaded and target_name in ["Pq", "Vq", "Pk", "Vk", "Pv", "Vv"]:
                    print(f"[warn] Layer {i} missing parameter: {target_name}")

            # Load LayerNorms
            ln1_weight_key = f"{layer_prefix}ln1.weight"
            ln1_bias_key = f"{layer_prefix}ln1.bias"
            ln2_weight_key = f"{layer_prefix}ln2.weight"
            ln2_bias_key = f"{layer_prefix}ln2.bias"

            if all(k in state_dict for k in [ln1_weight_key, ln1_bias_key]):
                svd_block.ln1 = nn.LayerNorm(state_dict[ln1_weight_key].shape[0])
                svd_block.ln1.weight = nn.Parameter(state_dict[ln1_weight_key])
                svd_block.ln1.bias = nn.Parameter(state_dict[ln1_bias_key])
                loaded_params += 2

            if all(k in state_dict for k in [ln2_weight_key, ln2_bias_key]):
                svd_block.ln2 = nn.LayerNorm(state_dict[ln2_weight_key].shape[0])
                svd_block.ln2.weight = nn.Parameter(state_dict[ln2_weight_key])
                svd_block.ln2.bias = nn.Parameter(state_dict[ln2_bias_key])
                loaded_params += 2

            # Wrap in shim
            encoder_layers[i] = BertLayerShim(svd_block)

    print(f"[info] Loaded {loaded_params} SVD parameters across {num_layers} layers")

    # Load remaining model parameters (embeddings, classifier, etc.)
    print("[load] Loading remaining model parameters...")

    # Filter state_dict to only include non-encoder-layer keys
    remaining_state = {
        k: v for k, v in state_dict.items()
        if f"{encoder_prefix}." not in k or ".block." not in k
    }

    missing_keys, unexpected_keys = base_model.load_state_dict(remaining_state, strict=False)

    if missing_keys:
        print(f"[warn] Missing keys: {len(missing_keys)}")
        if len(missing_keys) <= 10:
            for k in missing_keys:
                print(f"  - {k}")

    # Move to device and dtype
    base_model = base_model.to(device=device, dtype=dtype).eval()

    # Enable FlashSVD backend if specified
    if comp_info.get('backend') == 'flashsvd' and comp_info.get('method') != 'dense':
        print(f"[load] Enabling FlashSVD backend...")
        try:
            from src.encoders.backend import enable_flashsvd
            enable_flashsvd(base_model)
            print(f"[load] FlashSVD backend enabled successfully")
        except Exception as e:
            print(f"[warn] Failed to enable FlashSVD backend: {e}")
            print(f"[warn] Falling back to naive backend")

    print(f"[load] Model loaded successfully")
    return base_model, tokenizer, comp_info


class MinimalSVDBlock(nn.Module):
    """
    Minimal SVD block structure that matches NaiveSVDBlock parameter layout.

    This class defines the forward pass and parameter structure, but doesn't
    initialize parameters. Instead, parameters are loaded from state_dict.
    """

    def __init__(self):
        super().__init__()
        # Parameters will be loaded from state_dict
        # We just need to define the structure here
        self.attn_mode = "einsum"  # "einsum" | "sdpa" — patchable after load

    def forward(self, x, mask=None):
        import math

        B, M, dm = x.shape

        # Check if parameters exist
        if not hasattr(self, 'Pq'):
            raise RuntimeError("SVD parameters not loaded! Use load_state_dict first.")

        # Handle three parameter layouts:
        # 1. Full-matrix mode: 2D [dm, R] - naive backend with full-matrix SVD
        # 2. Per-head mode (FlashSVD): 3D [H, dh, R]
        # 3. Per-head mode (Naive): 4D [1, H, dm, R]  — 3rd dim is d_model, not head_dim!

        if self.Pq.ndim == 2:
            # Full-matrix mode: [dm, R]
            # Q = x @ Uq @ Vq^T + bq
            _, R = self.Pq.shape
            H = getattr(self, 'num_heads', 12)  # Injected at load time; fallback 12 for BERT-base
            dh = dm // H
            scale = 1.0 / math.sqrt(dh)

            # Full-matrix projection: x [B, M, dm] @ U [dm, R] @ V [R, dm] + bias [dm]
            def project_full(x, U, V, b):
                """Full-matrix SVD: W = U @ V, so x @ W = x @ U @ V"""
                tmp = torch.matmul(x, U)  # [B, M, dm] @ [dm, R] = [B, M, R]
                out = torch.matmul(tmp, V)  # [B, M, R] @ [R, dm] = [B, M, dm]
                if b is not None:
                    out = out + b.unsqueeze(0).unsqueeze(0)  # Broadcast bias
                return out

            # Project to Q, K, V (all [B, M, dm])
            Q_flat = project_full(x, self.Pq, self.Vq, self.bq if hasattr(self, 'bq') else None)
            K_flat = project_full(x, self.Pk, self.Vk, self.bk if hasattr(self, 'bk') else None)
            V_flat = project_full(x, self.Pv, self.Vv, self.bv if hasattr(self, 'bv') else None)

            # Reshape to multi-head: [B, M, dm] -> [B, H, M, dh]
            Q = Q_flat.view(B, M, H, dh).transpose(1, 2)  # [B, H, M, dh]
            K = K_flat.view(B, M, H, dh).transpose(1, 2)
            V = V_flat.view(B, M, H, dh).transpose(1, 2)

        elif self.Pq.ndim == 3:
            # FlashSVD backend: [H, dh, R]
            H, dh, R = self.Pq.shape
            scale = 1.0 / math.sqrt(dh)
            Pq, Vq, Pk, Vk, Pv, Vv = self.Pq, self.Vq, self.Pk, self.Vk, self.Pv, self.Vv

            def project(x, P, V, b):
                tmp = torch.einsum("bmd,hdr->bhmr", x, P)
                out = torch.einsum("bhmr,hrd->bhmd", tmp, V)
                if b is not None:
                    out = out + b
                return out

            Q = project(x, Pq, Vq, self.bq if hasattr(self, 'bq') else None)
            K = project(x, Pk, Vk, self.bk if hasattr(self, 'bk') else None)
            V = project(x, Pv, Vv, self.bv if hasattr(self, 'bv') else None)

        elif self.Pq.ndim == 4:
            # Naive backend: [1, H, d_model, R] — note the 3rd dim is d_model, NOT head_dim
            _, H, _, R = self.Pq.shape
            dh = dm // H  # Correct head dimension (e.g. 64 for BERT-base)
            scale = 1.0 / math.sqrt(dh)
            Pq, Vq, Pk, Vk, Pv, Vv = self.Pq[0], self.Vq[0], self.Pk[0], self.Vk[0], self.Pv[0], self.Vv[0]

            def project(x, P, V, b):
                tmp = torch.einsum("bmd,hdr->bhmr", x, P)
                out = torch.einsum("bhmr,hrd->bhmd", tmp, V)
                if b is not None:
                    out = out + b
                return out

            Q = project(x, Pq, Vq, self.bq if hasattr(self, 'bq') else None)
            K = project(x, Pk, Vk, self.bk if hasattr(self, 'bk') else None)
            V = project(x, Pv, Vv, self.bv if hasattr(self, 'bv') else None)

        else:
            raise ValueError(f"Unexpected Pq shape: {self.Pq.shape}. Expected 2D [dm, R], 3D [H, dh, R], or 4D [1, H, dm, R]")

        # Attention kernel: einsum (paper-faithful O(n²)) or sdpa (PyTorch flash-attn)
        import torch.nn.functional as _F
        if getattr(self, 'attn_mode', 'einsum') == 'sdpa':
            sdpa_mask = mask.view(B, 1, 1, M).to(torch.bool) if mask is not None else None
            attn = _F.scaled_dot_product_attention(Q, K, V, attn_mask=sdpa_mask, scale=scale, dropout_p=0.0)
        else:
            logits = torch.einsum("bhmd,bhnd->bhmn", Q, K) * scale
            if mask is not None:
                m = mask.view(B, 1, 1, M).to(torch.bool)
                logits = logits.masked_fill(~m, torch.finfo(logits.dtype).min)
            A = torch.softmax(logits, dim=-1)
            attn = torch.einsum("bhmn,bhnd->bhmd", A, V)

        # Output projection + LN
        attn = attn.transpose(1, 2).reshape(B, M, dm)
        x1 = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn)

        # FFN
        import torch.nn.functional as F
        mid = x1 @ self.U1
        midV = mid @ self.V1
        midA = F.gelu(midV + self.b1)
        y = (midA @ self.U2) @ self.V2 + self.b2
        return self.ln2(x1 + y)


class MinimalModernBertSVDBlock(nn.Module):
    """ModernBERT SVD block for load_compressed_model.

    Mirrors NaiveModernBertSVDBlock.forward() exactly.
    All parameters and sub-modules are injected at load time from the
    checkpoint state dict (SVD params, LayerNorms, Wo_attn) and base
    model (rotary_emb).
    """

    def __init__(self):
        super().__init__()
        # Injected at load time — no parameters created in __init__

    def forward(self, hidden_states, attention_mask=None,
                sliding_window_mask=None, position_ids=None,
                output_attentions=False, **kwargs):
        import torch.nn.functional as _F

        B, M, D = hidden_states.shape
        H, dh = self.num_heads, self.head_dim
        x = hidden_states

        # Pre-norm
        xn = self.attn_norm(x)

        def project(xn, P, V, b):
            tmp = torch.einsum("bmd,hdr->bhmr", xn, P)
            out = torch.einsum("bhmr,hrd->bhmd", tmp, V)
            if b is not None:
                out = out + b.view(1, H, 1, dh)
            return out  # [B, H, M, dh]

        Q = project(xn, self.Pq, self.Vq, getattr(self, 'bq', None))
        K = project(xn, self.Pk, self.Vk, getattr(self, 'bk', None))
        V = project(xn, self.Pv, self.Vv, getattr(self, 'bv', None))

        # RoPE
        if position_ids is None:
            position_ids = torch.arange(M, device=x.device).unsqueeze(0).expand(B, M)
        qf = Q.reshape(B * H, M, dh)
        kf = K.reshape(B * H, M, dh)
        posf = position_ids.unsqueeze(1).expand(B, H, M).reshape(B * H, M)
        try:
            cos, sin = self.rotary_emb(qf, position_ids=posf, layer_type=getattr(self, 'attention_type', 'global'))
        except (TypeError, KeyError):
            cos, sin = self.rotary_emb(qf, position_ids=posf)
        Q = _apply_rotary(qf, cos, sin).view(B, H, M, dh)
        K = _apply_rotary(kf, cos, sin).view(B, H, M, dh)

        # SDPA mask (same logic as NaiveModernBertSVDBlock)
        sdpa_mask = None
        if sliding_window_mask is not None:
            sm = sliding_window_mask
            if sm.dtype.is_floating_point and sm.dtype != Q.dtype:
                sm = sm.to(Q.dtype)
            sdpa_mask = sm
        elif attention_mask is not None:
            if attention_mask.dim() == 2:
                sdpa_mask = ~(attention_mask.to(torch.bool))[:, None, None, :]
            elif attention_mask.dim() == 4:
                sm = attention_mask
                if sm.dtype.is_floating_point and sm.dtype != Q.dtype:
                    sm = sm.to(Q.dtype)
                sdpa_mask = sm

        attn = _F.scaled_dot_product_attention(Q, K, V, attn_mask=sdpa_mask, dropout_p=0.0)
        attn = attn.transpose(1, 2).reshape(B, M, D)
        attn_out = (attn @ self.Uo) @ self.Vo
        if getattr(self, 'bo_attn', None) is not None:
            attn_out = attn_out + self.bo_attn
        x = x + attn_out

        # FFN (pre-norm, GeGLU or GELU)
        xn2 = self.mlp_norm(x)
        z = (xn2 @ self.U1) @ self.V1
        b1 = getattr(self, 'b1', None)
        if b1 is not None:
            z = z + b1

        if self.ffn_is_geglu:
            z1, z2 = z.chunk(2, dim=-1)
            h = _F.gelu(z1, approximate=self.gelu_approximate) * z2
        else:
            h = _F.gelu(z, approximate=self.gelu_approximate)

        y = (h @ self.U2) @ self.V2
        b2 = getattr(self, 'b2', None)
        if b2 is not None:
            y = y + b2
        x = x + y

        if output_attentions:
            return (x, None)
        return x


def test_loading():
    """Test loading a compressed model."""
    checkpoint = "compressed_models/bert/fwsvd/fwsvd_r300_naive"

    model, tokenizer, info = load_compressed_model(
        checkpoint,
        device="cuda",
        dtype=torch.float32,
    )

    print(f"\n[test] Model loaded successfully!")
    print(f"[test] Expected accuracy: {info['accuracy_before_finetune']:.4f}")

    # Test forward pass with full validation
    from datasets import load_dataset
    from evaluate import load as load_metric

    dataset = load_dataset("glue", "sst2", split="validation")
    metric = load_metric("accuracy")

    print(f"\n[test] Running validation on {len(dataset)} examples...")

    batch_size = 32
    correct = 0
    total = 0

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        inputs = tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

        metric.add_batch(
            predictions=preds.cpu(),
            references=torch.tensor(batch["label"])
        )
        correct += (preds.cpu() == torch.tensor(batch["label"])).sum().item()
        total += len(batch["label"])

    accuracy = metric.compute()["accuracy"]
    print(f"[test] Validation accuracy: {accuracy:.4f}")
    print(f"[test] Expected accuracy: {info['accuracy_before_finetune']:.4f}")
    print(f"[test] Match: {'✓' if abs(accuracy - info['accuracy_before_finetune']) < 0.001 else '✗'}")


if __name__ == "__main__":
    test_loading()
