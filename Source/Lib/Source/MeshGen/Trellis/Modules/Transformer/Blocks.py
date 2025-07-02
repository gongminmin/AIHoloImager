# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/modules/attention/blocks.py

from typing import *

import torch
import torch.nn as nn

from ..Norm import LayerNorm32

class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """

    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()

        self.channels = channels
        self.in_channels = in_channels
        freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(freq_dim, dtype = torch.float32) / freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)

    def SinCosEmbedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.

        Args:
            x: a 1-D Tensor of N indices

        Returns:
            an (N, D) Tensor of positional embeddings.
        """

        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim = -1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, D) tensor of spatial positions
        """

        num, dim = x.shape
        assert dim == self.in_channels, "Input dimension must match number of input channels"
        embed = self.SinCosEmbedding(x.reshape(-1))
        embed = embed.reshape(num, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat((embed, torch.zeros(num, self.channels - embed.shape[1], device = embed.device)), dim = -1)
        return embed

class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0, device: Optional[torch.device] = None):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio), device = device),
            nn.GELU(approximate = "tanh"),
            nn.Linear(int(channels * mlp_ratio), channels, device = device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
