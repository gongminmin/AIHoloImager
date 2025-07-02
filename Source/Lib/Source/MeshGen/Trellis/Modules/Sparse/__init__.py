# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/modules/sparse/__init__.py

import importlib
from typing import *

__attributes = {
    "SparseTensor": "Basic",
    "SparseBatchBroadcast": "Basic",
    "SparseUnbind": "Basic",
    "SparseGroupNorm": "Norm",
    "SparseGroupNorm32": "Norm",
    "SparseSiLU": "Nonlinearity",
    "SparseGELU": "Nonlinearity",
    "SparseLinear": "Linear",
    "SparseScaledDotProductAttention": "Attention",
    "SparseWindowedScaledDotProductSelfAttention": "Attention",
    "SparseMultiHeadAttention": "Attention",
    "SparseConv3D": "Conv",
    "SparseDownsample": "Spatial",
    "SparseUpsample": "Spatial",
    "SparseSubdivide" : "Spatial"
}

__submodules = ["Transformer"]

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]
