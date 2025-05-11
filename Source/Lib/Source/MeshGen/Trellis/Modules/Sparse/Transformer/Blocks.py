# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/modules/sparse/transformer/blocks.py

from typing import *

import torch
import torch.nn as nn

from ..Attention import SparseMultiHeadAttention, SerializeMode
from ..Basic import SparseTensor
from ..Linear import SparseLinear
from ..Nonlinearity import SparseGELU
from ...Norm import LayerNorm32

class SparseFeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0, device : Optional[torch.device] = None):
        super().__init__()

        self.mlp = nn.Sequential(
            SparseLinear(channels, int(channels * mlp_ratio), device = device),
            SparseGELU(approximate="tanh"),
            SparseLinear(int(channels * mlp_ratio), channels, device = device),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.mlp(x)

class SparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN).
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
        device : Optional[torch.device] = None,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6, device = device)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6, device = device)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
            device = device,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
            device = device,
        )

    def _forward(self, x: SparseTensor) -> SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = self.attn(h)
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: SparseTensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)
