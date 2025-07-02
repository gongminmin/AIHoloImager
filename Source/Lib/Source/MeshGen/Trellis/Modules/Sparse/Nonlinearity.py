# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/modules/sparse/nonlinearity.py

import torch.nn as nn

from . import SparseTensor

__all__ = [
    "SparseSiLU",
    "SparseGELU",
]

class SparseSiLU(nn.SiLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))

class SparseGELU(nn.GELU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))
