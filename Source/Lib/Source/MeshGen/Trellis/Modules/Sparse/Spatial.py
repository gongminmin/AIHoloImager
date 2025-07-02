# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/modules/sparse/spatial.py

from typing import *

import torch
import torch.nn as nn

from . import SparseTensor

__all__ = [
    "SparseDownsample",
    "SparseUpsample",
    "SparseSubdivide",
]

class SparseDownsample(nn.Module):
    """
    Downsample a sparse tensor by a factor of `factor`.
    Implemented as average pooling.
    """

    def __init__(self, factor: Union[int, Tuple[int, ...], List[int]]):
        super(SparseDownsample, self).__init__()
        self.factor = tuple(factor) if isinstance(factor, (list, tuple)) else factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        dim = input.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * dim
        assert dim == len(factor), 'Input coordinates must have the same dimension as the downsample factor.'

        coord = list(input.coords.unbind(dim = -1))
        for i, f in enumerate(factor):
            coord[i + 1] = coord[i + 1] // f

        max_items = [coord[i + 1].max().item() + 1 for i in range(dim)]
        offset = torch.cumprod(torch.tensor(max_items[:: -1]), 0).tolist()[:: -1] + [1]
        code = sum([c * o for c, o in zip(coord, offset)])
        code, idx = code.unique(return_inverse = True)

        new_feats = torch.scatter_reduce(
            torch.zeros(code.shape[0], input.feats.shape[1], device = input.feats.device, dtype = input.feats.dtype),
            dim = 0,
            index = idx.unsqueeze(1).expand(-1, input.feats.shape[1]),
            src = input.feats,
            reduce = "mean"
        )
        new_coords = torch.stack(
            [code // offset[0]] +
            [(code // offset[i + 1]) % max_items[i] for i in range(dim)],
            dim = -1
        )
        out = SparseTensor(new_feats, new_coords, input.shape)
        out._scale = tuple([s // f for s, f in zip(input._scale, factor)])
        out._spatial_cache = input._spatial_cache

        out.RegisterSpatialCache(f"upsample_{factor}_coords", input.coords)
        out.RegisterSpatialCache(f"upsample_{factor}_layout", input.layout)
        out.RegisterSpatialCache(f"upsample_{factor}_idx", idx)

        return out

class SparseUpsample(nn.Module):
    """
    Upsample a sparse tensor by a factor of `factor`.
    Implemented as nearest neighbor interpolation.
    """

    def __init__(self, factor: Union[int, Tuple[int, int, int], List[int]]):
        super(SparseUpsample, self).__init__()
        self.factor = tuple(factor) if isinstance(factor, (list, tuple)) else factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        dim = input.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor, ) * dim
        assert dim == len(factor), 'Input coordinates must have the same dimension as the upsample factor.'

        new_coords = input.GetSpatialCache(f"upsample_{factor}_coords")
        new_layout = input.GetSpatialCache(f"upsample_{factor}_layout")
        idx = input.GetSpatialCache(f"upsample_{factor}_idx")
        if any([x is None for x in [new_coords, new_layout, idx]]):
            raise ValueError("Upsample cache not found. SparseUpsample must be paired with SparseDownsample.")
        new_feats = input.feats[idx]
        out = SparseTensor(new_feats, new_coords, input.shape, new_layout)
        out._scale = tuple([s * f for s, f in zip(input._scale, factor)])
        out._spatial_cache = input._spatial_cache
        return out

class SparseSubdivide(nn.Module):
    """
    Upsample a sparse tensor by a factor of `factor`.
    Implemented as nearest neighbor interpolation.
    """

    def __init__(self):
        super(SparseSubdivide, self).__init__()

    def forward(self, input: SparseTensor) -> SparseTensor:
        dim = input.coords.shape[-1] - 1
        n_cube = torch.ones([2] * dim, device = input.device, dtype = torch.int)
        n_coords = torch.nonzero(n_cube)
        n_coords = torch.cat([torch.zeros_like(n_coords[:, : 1]), n_coords], dim = -1)
        factor = n_coords.shape[0]
        assert factor == 2 ** dim
        new_coords = input.coords.clone()
        new_coords[:, 1 :] *= 2
        new_coords = new_coords.unsqueeze(1) + n_coords.unsqueeze(0).to(new_coords.dtype)

        new_feats = input.feats.unsqueeze(1).expand(input.feats.shape[0], factor, *input.feats.shape[1:])
        out = SparseTensor(new_feats.flatten(0, 1), new_coords.flatten(0, 1), input.shape)
        out._scale = input._scale * 2
        out._spatial_cache = input._spatial_cache
        return out
