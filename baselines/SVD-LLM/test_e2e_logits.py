#!/usr/bin/env python3
"""
End-to-end logit comparison: attention kernel vs fallback.
Loads compressed model, runs one short sequence with both paths,
checks if logits agree. If they agree, attention kernel is correct
and PPL=207194 must be caused by something else (MLP kernel).

Usage:
  python test_e2e_logits.py --model_path checkpoints/jeffwan_llama_7b_hf_whitening_only_0.5.pt
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
args = parser.parse_args()

from utils.model_utils import get_model_from_local
from flashsvd_component.svd_llama import SVD_LlamaAttention

print(f"Loading {args.model_path} ...")
model, tokenizer = get_model_from_local(args.model_path)
model.eval()
dev = torch.device("cuda")
model = model.to(dev)

# Short test input (avoid tokenizer quirks — just use random token ids)
torch.manual_seed(0)
input_ids = torch.randint(0, model.config.vocab_size, (1, 64), device=dev)

def set_attn_flashsvd(model, enabled: bool):
    """Monkey-patch use_flashsvd flag into every SVD_LlamaAttention."""
    for m in model.modules():
        if isinstance(m, SVD_LlamaAttention):
            m._force_flashsvd = enabled

def patched_forward(self, hidden_states, **kwargs):
    """Forward that respects _force_flashsvd override."""
    enabled = getattr(self, '_force_flashsvd', None)
    if enabled is None:
        enabled = hidden_states.is_cuda
    # Temporarily override
    orig = hidden_states.is_cuda  # always True on CUDA
    # We'll monkeypatch use_flashsvd directly by duplicating the relevant logic
    # Just redirect to original forward but with use_flashsvd controlled
    B, S, _ = hidden_states.size()
    H, dh   = self.num_heads, self.head_dim
    R       = self.q_v_proj.out_features

    past_key_value  = kwargs.get('past_key_value', None)
    past_key_values = kwargs.get('past_key_values', None)
    use_cache       = kwargs.get('use_cache', False)
    if past_key_value is None and past_key_values is not None:
        past_key_value = past_key_values

    if past_key_value is not None or use_cache:
        # decode path: always fallback
        import torch.nn.functional as F
        Q = (self.q_v_proj(hidden_states) @ self._eff_weight(self.q_u_proj).t()).view(B,S,H,dh).transpose(1,2)
        K_all = (self.k_v_proj(hidden_states) @ self._eff_weight(self.k_u_proj).t()).view(B,S,H,dh).transpose(1,2)
        V_all = (self.v_v_proj(hidden_states) @ self._eff_weight(self.v_u_proj).t()).view(B,S,H,dh).transpose(1,2)
        attn_out = F.scaled_dot_product_attention(Q, K_all, V_all, is_causal=(S>1))
        attn_output = attn_out.transpose(1,2).reshape(B,S,H*dh)
        attn_output = self.o_u_proj(self.o_v_proj(attn_output))
        return attn_output, None, None

    if not enabled:
        # Fallback path
        import torch.nn.functional as F
        from flashsvd_component.svd_llama import apply_rotary_pos_emb
        Q = self.q_u_proj(self.q_v_proj(hidden_states)).view(B,S,H,dh).transpose(1,2)
        K = self.k_u_proj(self.k_v_proj(hidden_states)).view(B,S,H,dh).transpose(1,2)
        V = self.v_u_proj(self.v_v_proj(hidden_states)).view(B,S,H,dh).transpose(1,2)
        cos_4d, sin_4d = self.rotary_emb(V, seq_len=S)
        pos = torch.arange(S, device=hidden_states.device).unsqueeze(0).expand(B,S)
        Q, K = apply_rotary_pos_emb(Q, K, cos_4d, sin_4d, pos)
        attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        attn_output = attn_out.transpose(1,2).reshape(B,S,H*dh)
    else:
        # Kernel path
        from src.kernels.decoder.flashsvdropeattn_v16 import (
            PackedFactors, build_rope_tables, flashsvd_attn_packed)
        Vq = self._eff_weight(self.q_u_proj).view(H,dh,R).permute(0,2,1).contiguous()
        Vk = self._eff_weight(self.k_u_proj).view(H,dh,R).permute(0,2,1).contiguous()
        Vv = self._eff_weight(self.v_u_proj).view(H,dh,R).permute(0,2,1).contiguous()
        Pq = self.q_v_proj(hidden_states).unsqueeze(2).expand(B,S,H,R)
        Pk = self.k_v_proj(hidden_states).unsqueeze(2).expand(B,S,H,R)
        Pv = self.v_v_proj(hidden_states).unsqueeze(2).expand(B,S,H,R)
        cos, sin = self._get_rope_tables(S, hidden_states.device, hidden_states.dtype)
        f = PackedFactors(Pq=Pq, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv)
        attn_bmhsd = flashsvd_attn_packed(f, cos, sin, causal=True)
        attn_output = attn_bmhsd.reshape(B,S,H*dh)

    attn_output = self.o_u_proj(self.o_v_proj(attn_output))
    return attn_output, None, None

# Patch all SVD_LlamaAttention modules
import types
for m in model.modules():
    if isinstance(m, SVD_LlamaAttention):
        m.forward = types.MethodType(patched_forward, m)

def run(label, enabled):
    set_attn_flashsvd(model, enabled)
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits  # [1, 64, vocab]
    return logits

print("Running fallback (attention kernel OFF)...")
logits_fb = run("fallback", False)

print("Running kernel  (attention kernel ON) ...")
logits_k  = run("kernel",   True)

diff = (logits_k - logits_fb).abs()
rel  = (diff / (logits_fb.abs() + 1e-6)).mean().item()
ma   = diff.max().item()
rf   = (diff.norm() / (logits_fb.norm() + 1e-12)).item()
fin  = torch.isfinite(logits_k).all().item()

status = "PASS" if (fin and rf < 1e-1) else "FAIL"
print(f"\n[{status}] logits finite={fin}  max_abs={ma:.3e}  rel_fro={rf:.3e}  mean_rel={rel:.3e}")

# PPL proxy: cross-entropy loss on random targets
targets = input_ids[:, 1:].contiguous()
shift_logits_fb = logits_fb[:, :-1, :].contiguous()
shift_logits_k  = logits_k[:, :-1, :].contiguous()
import torch.nn.functional as F
loss_fb = F.cross_entropy(shift_logits_fb.view(-1, model.config.vocab_size),
                          targets.view(-1)).item()
loss_k  = F.cross_entropy(shift_logits_k.view(-1, model.config.vocab_size),
                          targets.view(-1)).item()
print(f"  CE loss  fallback={loss_fb:.4f}  kernel={loss_k:.4f}  "
      f"(PPL fallback={torch.exp(torch.tensor(loss_fb)):.2f}  "
      f"kernel={torch.exp(torch.tensor(loss_k)):.2f})")
print()
