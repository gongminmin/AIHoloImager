# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/models/sparse_structure_vae.py

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as functional

from ..Modules.Norm import GroupNorm32, ChannelLayerNorm32
from ..Modules.Spatial import PixelShuffle3D
from ..Modules.Utils import ConvertModuleToFp16, ZeroModule

def NormLayer(norm_type: str, *args, **kwargs) -> nn.Module:
    if norm_type == "group":
        return GroupNorm32(32, *args, **kwargs)
    elif norm_type == "layer":
        return ChannelLayerNorm32(*args, **kwargs)
    else:
        raise ValueError(f"Invalid norm type {norm_type}")

class ResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type: Literal["group", "layer"] = "layer",
        device : Optional[torch.device] = None,
    ):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = NormLayer(norm_type, channels, device = device)
        self.norm2 = NormLayer(norm_type, self.out_channels, device = device)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding = 1, device = device)
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, 3, padding = 1, device = device)
        if device != "meta":
            self.conv = ZeroModule(self.conv)
        self.skip_connection = nn.Conv3d(channels, self.out_channels, 1, device = device) if channels != self.out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = functional.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = functional.silu(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h

class DownsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "avgpool"] = "conv",
        device : Optional[torch.device] = None,
    ):
        super().__init__()

        assert mode in ["conv", "avgpool"], f"Invalid mode {mode}"

        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels, 2, stride = 2, device = device)
        elif mode == "avgpool":
            assert in_channels == out_channels, "Pooling mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            return self.conv(x)
        else:
            return functional.avg_pool3d(x, 2)

class UpsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "nearest"] = "conv",
        device : Optional[torch.device] = None,
    ):
        super().__init__()
        
        assert mode in ["conv", "nearest"], f"Invalid mode {mode}"

        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding = 1, device = device)
        elif mode == "nearest":
            assert in_channels == out_channels, "Nearest mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            x = self.conv(x)
            return PixelShuffle3D(x, 2)
        else:
            return functional.interpolate(x, scale_factor = 2, mode = "nearest")

class SparseStructureDecoder(nn.Module):
    """
    Decoder for Sparse Structure (\\mathcal{D}_S in the paper Sec. 3.3).
    
    Args:
        out_channels (int): Channels of the output.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the decoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        norm_type (Literal["group", "layer"]): Type of normalization layer.
        use_fp16 (bool): Whether to use FP16.
    """ 
    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
        device : Optional[torch.device] = None,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding = 1, device = device)

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], channels[0], device = device)
            for _ in range(num_res_blocks_middle)
        ])

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch, device = device)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks.append(
                    UpsampleBlock3d(ch, channels[i + 1], device = device)
                )

        self.out_layer = nn.Sequential(
            NormLayer(norm_type, channels[-1], device = device),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding = 1, device = device)
        )

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
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(ConvertModuleToFp16)
        self.middle_block.apply(ConvertModuleToFp16)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        
        h = h.type(self.dtype)
                
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)

        h = h.type(x.dtype)
        h = self.out_layer(h)
        return h
