# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/models/structured_latent_vae/decoder_mesh.py

from typing import *

import numpy as np
import torch
import torch.nn as nn

from ...Modules.Utils import ConvertModuleToFp16, ZeroModule
from ...Modules import Sparse as sp
from .Base import SparseTransformerBase

class SparseVolumeExtractResult:
    def __init__(self, resolution: int, x: sp.SparseTensor):
        self.resolution = resolution
        self.coords = x.coords[:, 1 :].to(torch.int32)
        self.feats = x.feats.to(torch.float16)

class SparseSubdivideBlock3D(nn.Module):
    """
    A 3D subdivide block that can subdivide the sparse tensor.

    Args:
        channels: channels in the inputs and outputs.
        out_channels: if specified, the number of output channels.
        num_groups: the number of groups for the group norm.
    """

    def __init__(
        self,
        channels: int,
        resolution: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32,
        device: Optional[torch.device] = None,
    ):
        super(SparseSubdivideBlock3D, self).__init__()

        out_resolution = resolution * 2
        if out_channels is None:
            out_channels = channels
        indice_key = f"res_{out_resolution}"

        self.act_layers = nn.Sequential(
            sp.SparseGroupNorm32(num_groups, channels, device = device),
            sp.SparseSiLU()
        )

        self.sub = sp.SparseSubdivide()

        self.out_layers = nn.Sequential(
            sp.SparseConv3D(channels, out_channels, 3, indice_key = indice_key),
            sp.SparseGroupNorm32(num_groups, out_channels, device = device),
            sp.SparseSiLU(),
            sp.SparseConv3D(out_channels, out_channels, 3, indice_key = indice_key),
        )
        if device != "meta":
            self.out_layers[3] = ZeroModule(self.out_layers[3])

        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sp.SparseConv3D(channels, out_channels, 1, indice_key = indice_key)

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: an [N x C x ...] Tensor of features.
        Returns:
            an [N x C x ...] Tensor of outputs.
        """

        h = self.act_layers(x)
        h = self.sub(h)
        x = self.sub(x)
        h = self.out_layers(h)
        h = h + self.skip_connection(x)
        return h

class SLatMeshDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        device: Optional[torch.device] = None,
    ):
        super(SLatMeshDecoder, self).__init__(
            in_channels = latent_channels,
            model_channels = model_channels,
            num_blocks = num_blocks,
            num_heads = num_heads,
            num_head_channels = num_head_channels,
            mlp_ratio = mlp_ratio,
            attn_mode = attn_mode,
            window_size = window_size,
            pe_mode = pe_mode,
            use_fp16 = use_fp16,
            qk_rms_norm = qk_rms_norm,
            device = device,
        )

        self.resolution = resolution
        out_channels = 8 * 1 + 8 * 3 + 21 + 8 * 6 # 8 densities, 8 deformation vectors, 21 weights, 8 colors, 8 normals
        self.upsample = nn.ModuleList([
            SparseSubdivideBlock3D(
                channels = model_channels,
                resolution = resolution,
                out_channels = model_channels // 4,
                device = device,
            ),
            SparseSubdivideBlock3D(
                channels = model_channels // 4,
                resolution = resolution * 2,
                out_channels = model_channels // 8,
                device = device,
            )
        ])
        self.out_layer = sp.SparseLinear(model_channels // 8, out_channels, device = device)

        if device != "meta":
            self.InitializeWeights()
        if use_fp16:
            self.ConvertToFp16()

    def InitializeWeights(self) -> None:
        super().InitializeWeights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def ConvertToFp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """

        super().ConvertToFp16()
        self.upsample.apply(ConvertModuleToFp16)
    
    def ToRepresentation(self, x: sp.SparseTensor) -> List[SparseVolumeExtractResult]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """

        ret = []
        for i in range(x.shape[0]):
            ret.append(SparseVolumeExtractResult(self.resolution * 4, x[i]))
        return ret

    def forward(self, x: sp.SparseTensor) -> List[SparseVolumeExtractResult]:
        h = super().forward(x)
        for block in self.upsample:
            h = block(h)
        h = h.type(x.dtype)
        h = self.out_layer(h)
        return self.ToRepresentation(h)
