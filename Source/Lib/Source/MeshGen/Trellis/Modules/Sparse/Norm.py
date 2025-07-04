# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/modules/sparse/norm.py

from typing import Optional

import torch
import torch.nn as nn

from . import SparseTensor

__all__ = [
    "SparseGroupNorm",
    "SparseGroupNorm32",
]

class SparseGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps = 1e-5, affine = True, device: Optional[torch.device] = None):
        super(SparseGroupNorm, self).__init__(num_groups, num_channels, eps, affine, device = device)

    def forward(self, input: SparseTensor) -> SparseTensor:
        nfeats = torch.zeros_like(input.feats)
        for k in range(input.shape[0]):
            bfeats = input.feats[input.layout[k]]
            bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
            nfeats[input.layout[k]] = bfeats
        return input.replace(nfeats)

class SparseGroupNorm32(SparseGroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return super().forward(x.float()).type(x.dtype)
