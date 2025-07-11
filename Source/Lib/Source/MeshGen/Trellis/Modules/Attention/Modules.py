# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/modules/attention/modules.py

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .FullAttn import ScaledDotProductAttention

class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int, device: Optional[torch.device] = None):
        super().__init__()

        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim, device = device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (functional.normalize(x.float(), dim = -1) * self.gamma * self.scale).to(x.dtype)

class RotaryPositionEmbedder(nn.Module):
    def __init__(self, hidden_size: int, in_channels: int = 3):
        super().__init__()

        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2"

        self.hidden_size = hidden_size
        freq_dim = hidden_size // in_channels // 2
        self.freqs = torch.arange(freq_dim, dtype = torch.float32) / freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)

    def GetPhases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases

    def RotaryEmbedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[: -1], -1, 2))
        x_rotated = x_complex * phases
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[: -1], -1).to(x.dtype)
        return x_embed

    def forward(self, q: torch.Tensor, k: torch.Tensor, indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (torch.Tensor): [..., N, D] tensor of queries
            k (torch.Tensor): [..., N, D] tensor of keys
            indices (torch.Tensor): [..., N, C] tensor of spatial positions
        """

        if indices is None:
            indices = torch.arange(q.shape[-2], device = q.device)
            if len(q.shape) > 2:
                indices = indices.unsqueeze(0).expand(q.shape[: -2] + (-1, ))

        phases = self.GetPhases(indices.reshape(-1)).reshape(*indices.shape[: -1], -1)
        if phases.shape[1] < self.hidden_size // 2:
            phases = torch.cat([phases, torch.polar(
                torch.ones(*phases.shape[: -1], self.hidden_size // 2 - phases.shape[1], device = phases.device),
                torch.zeros(*phases.shape[: -1], self.hidden_size // 2 - phases.shape[1], device = phases.device)
            )], dim = -1)
        q_embed = self.RotaryEmbedding(q, phases)
        k_embed = self.RotaryEmbedding(k, phases)
        return q_embed, k_embed
    
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"

        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")

        head_dim = channels // num_heads
        if ctx_channels is None:
            ctx_channels = channels
        self.num_heads = num_heads
        self.type = type
        self.attn_mode = attn_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self.type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias = qkv_bias, device = device)
        else:
            self.to_q = nn.Linear(channels, channels, bias = qkv_bias, device = device)
            self.to_kv = nn.Linear(ctx_channels, channels * 2, bias = qkv_bias, device = device)

        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(head_dim, num_heads, device = device)
            self.k_rms_norm = MultiHeadRMSNorm(head_dim, num_heads, device = device)

        self.to_out = nn.Linear(channels, channels, device = device)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, length, channels = x.shape
        if self.type == "self":
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(batch, length, 3, self.num_heads, -1)
            if self.use_rope:
                q, k, v = qkv.unbind(dim = 2)
                q, k = self.rope(q, k, indices)
                qkv = torch.stack([q, k, v], dim = 2)
            if self.attn_mode == "full":
                if self.qk_rms_norm:
                    q, k, v = qkv.unbind(dim = 2)
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                    h = ScaledDotProductAttention(q, k, v)
                else:
                    h = ScaledDotProductAttention(qkv)
            else:
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:
            lkv = context.shape[1]
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(batch, length, self.num_heads, -1)
            kv = kv.reshape(batch, lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim = 2)
                k = self.k_rms_norm(k)
                h = ScaledDotProductAttention(q, k, v)
            else:
                h = ScaledDotProductAttention(q, kv)
        h = h.reshape(batch, length, -1)
        h = self.to_out(h)
        return h
