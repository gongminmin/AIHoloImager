# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/models/structured_latent_vae/base.py

from typing import *

import torch
import torch.nn as nn

from ...Modules.Sparse.Transformer import SparseTransformerBlock
from ...Modules.Transformer import AbsolutePositionEmbedder
from ...Modules.Utils import ConvertModuleToFp16
from ...Modules import Sparse as sp

def BlockAttnConfig(self):
    """
    Return the attention configuration of the model.
    """
    for i in range(self.num_blocks):
        if self.attn_mode == "full":
            yield "full", None, None, None, None
        elif self.attn_mode == "swin":
            yield "windowed", self.window_size, None, self.window_size // 2 * (i % 2), None

class SparseTransformerBase(nn.Module):
    """
    Sparse Transformer without output layers.
    Serve as the base class for encoder and decoder.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        qk_rms_norm: bool = False,
        device : Optional[torch.device] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.qk_rms_norm = qk_rms_norm
        self.dtype = torch.float16 if use_fp16 else torch.float32

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels, device = device)
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode=attn_mode,
                window_size=window_size,
                shift_sequence=shift_sequence,
                shift_window=shift_window,
                serialize_mode=serialize_mode,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=self.qk_rms_norm,
                device = device,
            )
            for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in BlockAttnConfig(self)
        ])

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def ConvertToFp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(ConvertModuleToFp16)

    def InitializeWeights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.input_layer(x)
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(x.coords[:, 1:])
        h = h.type(self.dtype)
        for block in self.blocks:
            h = block(h)
        return h
