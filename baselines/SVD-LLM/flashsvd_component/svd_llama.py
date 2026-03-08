import math
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers import LlamaConfig

from src.kernels.decoder.flashsvdswiglu_v2 import flashsvd_ffn_dual_split_token
from src.kernels.decoder.flashsvdropeattn_v16 import (
    PackedFactors, DecodePackedFactors,
    build_rope_tables,
    flashsvd_attn_packed,
    flashsvd_attn_decode_packed,
)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SVD_LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        ratio=1
    ):
        super().__init__()
        self.ratio = ratio
        low_rank = int(intermediate_size * hidden_size * self.ratio / (intermediate_size + hidden_size))
        self.gate_u_proj = nn.Linear(low_rank, intermediate_size, bias=False)
        self.gate_v_proj = nn.Linear(hidden_size, low_rank, bias=False)
        
        self.down_u_proj = nn.Linear(low_rank, hidden_size, bias=False)
        self.down_v_proj = nn.Linear(intermediate_size, low_rank, bias=False)
        
        self.up_u_proj = nn.Linear(low_rank, intermediate_size, bias=False)
        self.up_v_proj = nn.Linear(hidden_size, low_rank, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        # Fast path: Triton FlashSVD SwiGLU on CUDA using dual P (independent gate/up)
        if x.is_cuda:
            PGate = self.gate_v_proj(x)           # [B, L, R]
            PUp   = self.up_v_proj(x)             # [B, L, R]
            GateU = self.gate_u_proj.weight.t()   # [R, D]
            UpU   = self.up_u_proj.weight.t()     # [R, D]
            DownV = self.down_v_proj.weight.t()   # [D, R]
            DownU = self.down_u_proj.weight.t()   # [R, H]
            return flashsvd_ffn_dual_split_token(PUp, PGate, GateU, UpU, DownV, DownU)

        # Fallback: baseline low-rank SwiGLU
        up = self.up_u_proj(self.up_v_proj(x))
        gate = self.gate_u_proj(self.gate_v_proj(x))
        return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))


class SVD_LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, ratio=1):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.ratio = ratio # 1 means no truncate, just keep normal attn

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        low_rank = int(self.hidden_size * self.ratio/2)
        self.q_u_proj = nn.Linear(low_rank, self.num_heads * self.head_dim, bias=False)
        self.q_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

        self.k_u_proj = nn.Linear(low_rank, self.num_heads * self.head_dim, bias=False)
        self.k_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

        self.v_u_proj = nn.Linear(low_rank, self.num_heads * self.head_dim, bias=False)
        self.v_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

        self.o_u_proj = nn.Linear(low_rank, self.hidden_size, bias=False)
        self.o_v_proj = nn.Linear(self.num_heads * self.head_dim, low_rank, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

        # Cached RoPE tables for v1.6 kernel: [max_len, dh/2]
        self._rope_cos: torch.Tensor = None
        self._rope_sin: torch.Tensor = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _get_rope_tables(self, seqlen: int, device, dtype):
        """Return (cos, sin) [seqlen, dh/2], rebuilding cache if needed."""
        if self._rope_cos is None or self._rope_cos.shape[0] < seqlen or \
                self._rope_cos.device != device or self._rope_cos.dtype != dtype:
            max_len = max(seqlen, self.max_position_embeddings)
            cos, sin = build_rope_tables(max_len, self.head_dim, 10000.0, device, dtype)
            self._rope_cos = cos
            self._rope_sin = sin
        return self._rope_cos[:seqlen], self._rope_sin[:seqlen]

    def _eff_weight(self, linear: nn.Module) -> torch.Tensor:
        """Return effective weight, merging LoRA delta if present."""
        W = linear.weight
        if hasattr(linear, 'lora_A') and hasattr(linear, 'lora_B'):
            adapter = getattr(linear, 'active_adapter', None)
            try:
                if adapter is not None and adapter in linear.lora_A and adapter in linear.lora_B:
                    W = W + (linear.lora_B[adapter].weight @ linear.lora_A[adapter].weight) * linear.scaling[adapter]
            except Exception:
                pass
        return W

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if past_key_value is None and past_key_values is not None:
            past_key_value = past_key_values

        B, S, _ = hidden_states.size()
        H, dh   = self.num_heads, self.head_dim
        R       = self.q_v_proj.out_features

        use_flashsvd = hidden_states.is_cuda

        # V factor matrices [H, R, dh] — shared across prefill and decode
        Vq = self._eff_weight(self.q_u_proj).view(H, dh, R).permute(0, 2, 1).contiguous()
        Vk = self._eff_weight(self.k_u_proj).view(H, dh, R).permute(0, 2, 1).contiguous()
        Vv = self._eff_weight(self.v_u_proj).view(H, dh, R).permute(0, 2, 1).contiguous()

        # ── Decode path (use_cache or incremental) ────────────────────────────
        if past_key_value is not None or use_cache:
            Pk_tok = self.k_v_proj(hidden_states).view(B, S, 1, R)
            Pv_tok = self.v_v_proj(hidden_states).view(B, S, 1, R)

            if past_key_value is not None:
                Pk_cache = torch.cat([past_key_value[0], Pk_tok], dim=1)
                Pv_cache = torch.cat([past_key_value[1], Pv_tok], dim=1)
            else:
                Pk_cache = Pk_tok
                Pv_cache = Pv_tok

            past_key_value = (Pk_cache, Pv_cache) if use_cache else None
            seqlen_k = int(Pk_cache.shape[1])

            if use_flashsvd:
                Pq_q = self.q_v_proj(hidden_states).view(B, S, R)[:, 0, :]
                Pq_q = Pq_q.unsqueeze(1).expand(B, H, R)
                Pk_exp = Pk_cache.expand(B, seqlen_k, H, R)
                Pv_exp = Pv_cache.expand(B, seqlen_k, H, R)
                cos, sin = self._get_rope_tables(seqlen_k, hidden_states.device, hidden_states.dtype)
                f_dec = DecodePackedFactors(
                    Pq=Pq_q.contiguous(), Pk=Pk_exp.contiguous(), Pv=Pv_exp.contiguous(),
                    Vq=Vq, Vk=Vk, Vv=Vv,
                )
                out_bhd = flashsvd_attn_decode_packed(f_dec, cos, sin, seqlen_k=seqlen_k, causal=True)
                attn_output = out_bhd.reshape(B, S, H * dh)
            else:
                # Fallback: reconstruct dense Q/K/V and run SDPA
                Q = (self.q_v_proj(hidden_states) @ self._eff_weight(self.q_u_proj).t()).view(B, S, H, dh).transpose(1, 2)
                K = Pk_cache.squeeze(2) @ self._eff_weight(self.k_u_proj).t()  # not quite right for cache
                # Simple fallback: re-expand cache
                K_all = Pk_cache.squeeze(2).view(B, seqlen_k, R) @ self._eff_weight(self.k_u_proj).t()
                V_all = Pv_cache.squeeze(2).view(B, seqlen_k, R) @ self._eff_weight(self.v_u_proj).t()
                K_all = K_all.view(B, seqlen_k, H, dh).transpose(1, 2)
                V_all = V_all.view(B, seqlen_k, H, dh).transpose(1, 2)
                import torch.nn.functional as F
                attn_out = F.scaled_dot_product_attention(Q, K_all, V_all, is_causal=(S > 1))
                attn_output = attn_out.transpose(1, 2).reshape(B, S, H * dh)

            attn_output = self.o_u_proj(self.o_v_proj(attn_output))
            return attn_output, None, past_key_value

        # ── Prefill path ───────────────────────────────────────────────────────
        if use_flashsvd:
            Pq = self.q_v_proj(hidden_states).unsqueeze(2).expand(B, S, H, R)
            Pk = self.k_v_proj(hidden_states).unsqueeze(2).expand(B, S, H, R)
            Pv = self.v_v_proj(hidden_states).unsqueeze(2).expand(B, S, H, R)
            cos, sin = self._get_rope_tables(S, hidden_states.device, hidden_states.dtype)
            f = PackedFactors(Pq=Pq, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv)
            attn_bmhsd = flashsvd_attn_packed(f, cos, sin, causal=True)
            attn_output = attn_bmhsd.reshape(B, S, H * dh)
        else:
            # Fallback: standard low-rank matmul + SDPA
            Q = self.q_u_proj(self.q_v_proj(hidden_states)).view(B, S, H, dh).transpose(1, 2)
            K = self.k_u_proj(self.k_v_proj(hidden_states)).view(B, S, H, dh).transpose(1, 2)
            V = self.v_u_proj(self.v_v_proj(hidden_states)).view(B, S, H, dh).transpose(1, 2)
            cos_4d, sin_4d = self.rotary_emb(V, seq_len=S)
            pos = torch.arange(S, device=hidden_states.device).unsqueeze(0).expand(B, S)
            Q, K = apply_rotary_pos_emb(Q, K, cos_4d, sin_4d, pos)
            import torch.nn.functional as F
            attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
            attn_output = attn_out.transpose(1, 2).reshape(B, S, H * dh)

        attn_output = self.o_u_proj(self.o_v_proj(attn_output))
        return attn_output, None, None
    
