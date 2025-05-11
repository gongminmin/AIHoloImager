# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/modules/sparse/linear.py

from typing import Optional

import torch
import torch.nn as nn

from . import SparseTensor

__all__ = [
    'SparseLinear'
]

class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device : Optional[torch.device] = None):
        super(SparseLinear, self).__init__(in_features, out_features, bias, device = device)

    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))
