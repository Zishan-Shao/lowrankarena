import math
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers import LlamaConfig

from kernels.flashsvdropeattn import FlashSVDRoPEAttention, QKVFactors
from kernels.flashsvdswiglu import flashsvd_ffn_swiglu


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
        # Fast path: Triton FlashSVD SwiGLU on CUDA using shared rank-space P
        if x.is_cuda:
            B, L, _ = x.shape
            R1 = self.up_v_proj.out_features
            D = self.up_u_proj.out_features

            # Rank-space input P via one low-rank projection (shared)
            P = self.up_v_proj(x)  # [B, L, R1]

            # Combine rank->intermediate factors for up and gate into V1 = [R1, 2D]
            V1u = self.up_u_proj.weight.t()    # [R1, D]
            V1v = self.gate_u_proj.weight.t()  # [R1, D]
            V1 = torch.cat([V1u, V1v], dim=1)

            # Down path factors
            U2 = self.down_v_proj.weight.t()   # [D,  R2]
            V2 = self.down_u_proj.weight.t()   # [R2, H]

            # Biases are absent in this module; pass zeros
            b1 = torch.zeros(2 * D, device=x.device, dtype=x.dtype)
            b2 = torch.zeros(V2.shape[1], device=x.device, dtype=x.dtype)

            y = flashsvd_ffn_swiglu(P, V1, U2, V2, b1, b2, use_autotune=True)
            return y

        # Fallback (CPU or non-CUDA): baseline low-rank SwiGLU
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

        # Flash SVD + RoPE attention kernel wrapper
        self.flash_attn = FlashSVDRoPEAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rotary_emb=self.rotary_emb,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        # HF >=4.40 passes `past_key_values`; accept it for compatibility
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if past_key_value is None and past_key_values is not None:
            past_key_value = past_key_values
        bsz, q_len, _ = hidden_states.size()

        if past_key_value is not None or use_cache:
            # The FlashSVDRoPEAttention kernel currently computes full-sequence attention
            # and does not implement KV caching. Fall back is not provided here.
            raise NotImplementedError("KV cache not supported with FlashSVDRoPEAttention in this path.")

        # Build low-rank P factors: [B, M, R] -> expand to [B, H, M, R]
        Pq = self.q_v_proj(hidden_states)  # [B, M, R]
        Pk = self.k_v_proj(hidden_states)  # [B, M, R]
        Pv = self.v_v_proj(hidden_states)  # [B, M, R]

        B, M, R = Pq.shape
        H, dh = self.num_heads, self.head_dim

        # Expand along heads (rank factors are shared across heads)
        # Expand across heads as views (zero stride on H) to avoid materialization
        Pq = Pq.unsqueeze(1).expand(B, H, M, R)
        Pk = Pk.unsqueeze(1).expand(B, H, M, R)
        Pv = Pv.unsqueeze(1).expand(B, H, M, R)

        # Build V factors from effective projection weights: [H, R, dh]
        # Include LoRA delta if adapters are active by reading lora_A/lora_B
        def _eff_weight(linear: nn.Module):
            W = linear.weight
            if hasattr(linear, 'lora_A') and hasattr(linear, 'lora_B'):
                adapter = getattr(linear, 'active_adapter', None)
                try:
                    if adapter is not None and adapter in linear.lora_A and adapter in linear.lora_B:
                        W = W + (linear.lora_B[adapter].weight @ linear.lora_A[adapter].weight) * linear.scaling[adapter]
                except Exception:
                    pass
            return W

        Vq = _eff_weight(self.q_u_proj).view(H, dh, R).permute(0, 2, 1).contiguous()
        Vk = _eff_weight(self.k_u_proj).view(H, dh, R).permute(0, 2, 1).contiguous()
        Vv = _eff_weight(self.v_u_proj).view(H, dh, R).permute(0, 2, 1).contiguous()

        # No biases in low-rank projections by default
        bq = bk = bv = None

        # Position ids default: [B, M]
        if position_ids is None:
            position_ids = torch.arange(M, device=hidden_states.device).unsqueeze(0).expand(B, M)

        # Attention mask handling: support 2D pad mask [B, M] or 4D additive [B,1,M,M]
        add_mask = None
        pad_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = attention_mask
            elif attention_mask.dim() == 4:
                if attention_mask.shape[-2] != M or attention_mask.shape[-1] != M:
                    raise NotImplementedError("Attention mask with differing q/k lengths not supported here.")
                add_mask = attention_mask
            else:
                raise ValueError(f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}")

        qkv = QKVFactors(Pq=Pq, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv, bq=bq, bk=bk, bv=bv)

        attn_bmhd = self.flash_attn(
            qkv,
            attention_mask=add_mask if add_mask is not None else pad_mask,
            position_ids=position_ids,
        )  # [B, M, H, dh]

        # Fold heads back to [B, M, D]
        attn_output = attn_bmhd.reshape(B, M, H * dh)

        # Low-rank output projection
        attn_output = self.o_u_proj(self.o_v_proj(attn_output))

        attn_weights = None
        past_key_value = None
        return attn_output, attn_weights, past_key_value
    
