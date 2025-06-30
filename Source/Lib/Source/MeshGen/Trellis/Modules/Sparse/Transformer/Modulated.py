# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/modules/sparse/transformer/modulated.py

from typing import *

import torch
import torch.nn as nn

from ..Basic import SparseTensor
from ..Attention import SparseMultiHeadAttention, SerializeMode
from ...Norm import LayerNorm32
from .Blocks import SparseFeedForwardNet

class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine = False, eps = 1e-6, device = device)
        self.norm2 = LayerNorm32(channels, elementwise_affine = True, eps = 1e-6, device = device)
        self.norm3 = LayerNorm32(channels, elementwise_affine = False, eps = 1e-6, device = device)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads = num_heads,
            type = "self",
            attn_mode = attn_mode,
            window_size = window_size,
            shift_sequence = shift_sequence,
            shift_window = shift_window,
            serialize_mode = serialize_mode,
            qkv_bias = qkv_bias,
            use_rope = use_rope,
            qk_rms_norm = qk_rms_norm,
            device = device,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels = ctx_channels,
            num_heads = num_heads,
            type = "cross",
            attn_mode = "full",
            qkv_bias = qkv_bias,
            qk_rms_norm = qk_rms_norm_cross,
            device = device,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio = mlp_ratio,
        )
        if not share_mod:
            self.ada_ln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias = True, device = device)
            )

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim = 1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln_modulation(mod).chunk(6, dim = 1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x
