# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/models/structured_latent_flow.py

from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from ..Modules.Norm import LayerNorm32
from ..Modules.Sparse.Transformer import ModulatedSparseTransformerCrossBlock
from ..Modules.Transformer import AbsolutePositionEmbedder
from ..Modules.Utils import ConvertModuleToFp16, ZeroModule
from ..Modules import Sparse as sp
from .SparseStructureFlow import TimestepEmbedder

class SparseResBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = channels

        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine = True, eps = 1e-6, device = device)
        self.norm2 = LayerNorm32(out_channels, elementwise_affine = False, eps = 1e-6, device = device)
        self.conv1 = sp.SparseConv3D(channels, out_channels, 3, device = device)
        self.conv2 = sp.SparseConv3D(out_channels, out_channels, 3, device = device)
        if device != "meta":
            self.conv2 = ZeroModule(self.conv2)
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * out_channels, bias = True, device = device),
        )
        self.skip_connection = sp.SparseLinear(channels, out_channels, device = device) if channels != out_channels else nn.Identity()
        if downsample:
            self.updown = sp.SparseDownsample(2)
        elif upsample:
            self.updown = sp.SparseUpsample(2)
        else:
            self.updown = None

    def UpDown(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.updown is not None:
            x = self.updown(x)
        return x

    def forward(self, x: sp.SparseTensor, emb: torch.Tensor) -> sp.SparseTensor:
        emb_out = self.emb_layers(emb).type(x.dtype)
        scale, shift = torch.chunk(emb_out, 2, dim = 1)

        x = self.UpDown(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(functional.silu(h.feats))
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(functional.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)

        return h

class SLatFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        if num_heads is None:
            num_heads = model_channels // num_head_channels
        self.pe_mode = pe_mode
        self.use_skip_connection = use_skip_connection
        self.share_mod = share_mod
        self.dtype = torch.float16 if use_fp16 else torch.float32

        assert int(np.log2(patch_size)) == np.log2(patch_size), "Patch size must be a power of 2"
        assert np.log2(patch_size) == len(io_block_channels), "Number of IO ResBlocks must match the number of stages"

        self.t_embedder = TimestepEmbedder(model_channels, device = device)
        if share_mod:
            self.ada_ln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias = True, device = device)
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, io_block_channels[0], device = device)
        self.input_blocks = nn.ModuleList([])
        for chs, next_chs in zip(io_block_channels, io_block_channels[1 :] + [model_channels]):
            self.input_blocks.extend([
                SparseResBlock3D(
                    chs,
                    model_channels,
                    out_channels = chs,
                    device = device,
                )
                for _ in range(num_io_res_blocks-1)
            ])
            self.input_blocks.append(
                SparseResBlock3D(
                    chs,
                    model_channels,
                    out_channels = next_chs,
                    downsample = True,
                    device = device,
                )
            )

        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads = num_heads,
                mlp_ratio = mlp_ratio,
                attn_mode = "full",
                use_rope = (pe_mode == "rope"),
                share_mod = self.share_mod,
                qk_rms_norm = qk_rms_norm,
                qk_rms_norm_cross = qk_rms_norm_cross,
                device = device,
            )
            for _ in range(num_blocks)
        ])

        self.out_blocks = nn.ModuleList([])
        for chs, prev_chs in zip(reversed(io_block_channels), [model_channels] + list(reversed(io_block_channels[1 :]))):
            self.out_blocks.append(
                SparseResBlock3D(
                    prev_chs * 2 if self.use_skip_connection else prev_chs,
                    model_channels,
                    out_channels = chs,
                    upsample = True,
                    device = device,
                )
            )
            self.out_blocks.extend([
                SparseResBlock3D(
                    chs * 2 if self.use_skip_connection else chs,
                    model_channels,
                    out_channels = chs,
                    device = device,
                )
                for _ in range(num_io_res_blocks - 1)
            ])
        self.out_layer = sp.SparseLinear(io_block_channels[0], out_channels, device = device)

        if device != "meta":
            self.InitializeWeights()
        if use_fp16:
            self.ConvertToFp16()

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

        self.input_blocks.apply(ConvertModuleToFp16)
        self.blocks.apply(ConvertModuleToFp16)
        self.out_blocks.apply(ConvertModuleToFp16)

    def InitializeWeights(self) -> None:
        # Initialize transformer layers:
        def BasicInit(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(BasicInit)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std = 0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std = 0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.ada_ln_modulation[-1].weight, 0)
            nn.init.constant_(self.ada_ln_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.ada_ln_modulation[-1].weight, 0)
                nn.init.constant_(block.ada_ln_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: torch.Tensor) -> sp.SparseTensor:
        h = self.input_layer(x).type(self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.ada_ln_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)

        skips = []
        # pack with input blocks
        for block in self.input_blocks:
            h = block(h, t_emb)
            skips.append(h.feats)

        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.coords[:, 1 :]).type(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)

        # unpack with output blocks
        for block, skip in zip(self.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim = 1)), t_emb)
            else:
                h = block(h, t_emb)

        h = h.replace(functional.layer_norm(h.feats, h.feats.shape[-1 :]))
        h = self.out_layer(h.type(x.dtype))
        return h
