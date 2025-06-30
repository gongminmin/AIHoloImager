# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/modules/sparse/attention/modules.py

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .. import SparseTensor
from .FullAttn import SparseScaledDotProductAttention
from .SerializedAttn import SerializeMode
from .WindowedAttn import SparseWindowedScaledDotProductSelfAttention
from ...Attention import RotaryPositionEmbedder

class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int, device: Optional[torch.device] = None):
        super().__init__()

        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim, device = device))

    def forward(self, x: Union[SparseTensor, torch.Tensor]) -> Union[SparseTensor, torch.Tensor]:
        x_type = x.dtype
        x = x.float()
        if isinstance(x, SparseTensor):
            x = x.replace(functional.normalize(x.feats, dim = -1))
        else:
            x = functional.normalize(x, dim = -1)
        return (x * self.gamma * self.scale).to(x_type)

class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
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
        assert type == "self" or use_rope is False, "Rotary position embeddings only supported for self-attention"

        self.channels = channels
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_sequence = shift_sequence
        self.shift_window = shift_window
        self.serialize_mode = serialize_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias = qkv_bias, device = device)
        else:
            self.to_q = nn.Linear(channels, channels, bias = qkv_bias, device = device)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias = qkv_bias, device = device)

        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads, device = device)
            self.k_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads, device = device)

        self.to_out = nn.Linear(channels, channels, device = device)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    @staticmethod
    def Linear(module: nn.Linear, x: Union[SparseTensor, torch.Tensor]) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.replace(module(x.feats))
        else:
            return module(x)

    @staticmethod
    def ReshapeChs(x: Union[SparseTensor, torch.Tensor], shape: Tuple[int, ...]) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[: 2], *shape)

    def FusedPre(self, x: Union[SparseTensor, torch.Tensor], num_fused: int) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            x_feats = x.feats.unsqueeze(0)
        else:
            x_feats = x
        x_feats = x_feats.reshape(*x_feats.shape[: 2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0)) if isinstance(x, SparseTensor) else x_feats

    def Rope(self, qkv: SparseTensor) -> SparseTensor:
        q, k, v = qkv.feats.unbind(dim = 1)   # [T, H, C]
        q, k = self.rope(q, k, qkv.coords[:, 1 :])
        qkv = qkv.replace(torch.stack([q, k, v], dim = 1)) 
        return qkv

    def forward(self, x: Union[SparseTensor, torch.Tensor], context: Optional[Union[SparseTensor, torch.Tensor]] = None) -> Union[SparseTensor, torch.Tensor]:
        if self._type == "self":
            qkv = self.Linear(self.to_qkv, x)
            qkv = self.FusedPre(qkv, num_fused = 3)
            if self.use_rope:
                qkv = self.Rope(qkv)
            if self.qk_rms_norm:
                q, k, v = qkv.unbind(dim = 1)
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim = 1))
            if self.attn_mode == "full":
                h = SparseScaledDotProductAttention(qkv)
            elif self.attn_mode == "windowed":
                h = SparseWindowedScaledDotProductSelfAttention(
                    qkv, self.window_size, shift_window=self.shift_window
                )
        else:
            q = self.Linear(self.to_q, x)
            q = self.ReshapeChs(q, (self.num_heads, -1))
            kv = self.Linear(self.to_kv, context)
            kv = self.FusedPre(kv, num_fused = 2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim = 1)
                k = self.k_rms_norm(k)
                kv = kv.replace(torch.stack([k.feats, v.feats], dim = 1))
            h = SparseScaledDotProductAttention(q, kv)
        h = self.ReshapeChs(h, (-1, ))
        h = self.Linear(self.to_out, h)
        return h
