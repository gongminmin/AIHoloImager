# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/modules/sparse/nonlinearity.py

import torch.nn as nn

from . import SparseTensor

__all__ = [
    "SparseReLU",
    "SparseSiLU",
    "SparseGELU",
    "SparseActivatio",
]

class SparseReLU(nn.ReLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))

class SparseSiLU(nn.SiLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))

class SparseGELU(nn.GELU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))

class SparseActivation(nn.Module):
    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(self.activation(input.feats))
