# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/modules/sparse/conv/conv_spconv.py

import torch
import torch.nn as nn
import spconv.pytorch as spconv

from .. import SparseTensor

class SparseConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation = 1, bias = True, indice_key = None):
        super(SparseConv3D, self).__init__()

        self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, dilation = dilation, bias = bias, indice_key = indice_key)

    def forward(self, x: SparseTensor) -> SparseTensor:
        new_data = self.conv(x.data)
        new_shape = (x.shape[0], self.conv.out_channels)

        out = SparseTensor(
            new_data,
            shape = torch.Size(new_shape),
            layout = x.layout,
            scale = x._scale,
            spatial_cache = x._spatial_cache,
        )

        return out
