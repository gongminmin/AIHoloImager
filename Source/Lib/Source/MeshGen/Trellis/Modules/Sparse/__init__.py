# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/modules/sparse/__init__.py

from typing import *

import importlib

__attributes = {
    "SparseTensor": "Basic",
    "sparse_batch_broadcast": "Basic",
    "sparse_batch_op": "Basic",
    "sparse_cat": "Basic",
    "sparse_unbind": "Basic",
    "SparseGroupNorm": "Norm",
    "SparseLayerNorm": "Norm",
    "SparseGroupNorm32": "Norm",
    "SparseLayerNorm32": "Norm",
    "SparseReLU": "Nonlinearity",
    "SparseSiLU": "Nonlinearity",
    "SparseGELU": "Nonlinearity",
    "SparseActivation": "Nonlinearity",
    "SparseLinear": "Linear",
    "sparse_scaled_dot_product_attention": "Attention",
    "SerializeMode": "Attention",
    "sparse_windowed_scaled_dot_product_self_attention": "Attention",
    "SparseMultiHeadAttention": "Attention",
    "SparseConv3d": "Conv",
    "SparseInverseConv3d": "Conv",
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
