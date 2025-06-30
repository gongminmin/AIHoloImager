# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/modules/sparse/attention/windowed_attn.py

import math
from typing import *

import torch

from .. import SparseTensor
from ...Utils import MemEfficientAttention, BlockDiagonalMaskFromSeqlens

__all__ = [
    "SparseWindowedScaledDotProductSelfAttention",
]

def CalcWindowPartition(
    tensor: SparseTensor,
    window_size: Union[int, Tuple[int, ...]],
    shift_window: Union[int, Tuple[int, ...]] = 0
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    Calculate serialization and partitioning for a set of coordinates.

    Args:
        tensor (SparseTensor): The input tensor.
        window_size (int): The window size to use.
        shift_window (Tuple[int, ...]): The shift of serialized coordinates.

    Returns:
        (torch.Tensor): Forwards indices.
        (torch.Tensor): Backwards indices.
        (List[int]): Sequence lengths.
        (List[int]): Sequence batch indices.
    """

    dim = tensor.coords.shape[1] - 1
    shift_window = (shift_window,) * dim if isinstance(shift_window, int) else shift_window
    window_size = (window_size,) * dim if isinstance(window_size, int) else window_size
    shifted_coords = tensor.coords.clone().detach()
    shifted_coords[:, 1 :] += torch.tensor(shift_window, device = tensor.device, dtype = torch.int32).unsqueeze(0)

    max_coords = shifted_coords[:, 1 :].max(dim = 0).values.tolist()
    num_windows = [math.ceil((mc + 1) / ws) for mc, ws in zip(max_coords, window_size)]
    offset = torch.cumprod(torch.tensor([1] + num_windows[:: -1]), dim = 0).tolist()[:: -1]

    shifted_coords[:, 1 :] //= torch.tensor(window_size, device = tensor.device, dtype = torch.int32).unsqueeze(0)
    shifted_indices = (shifted_coords * torch.tensor(offset, device = tensor.device, dtype = torch.int32).unsqueeze(0)).sum(dim = 1)
    fwd_indices = torch.argsort(shifted_indices)
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device = tensor.device)
    seq_lens = torch.bincount(shifted_indices)
    seq_batch_indices = torch.arange(seq_lens.shape[0], device = tensor.device, dtype = torch.int32) // offset[0]
    mask = seq_lens != 0
    seq_lens = seq_lens[mask].tolist()
    seq_batch_indices = seq_batch_indices[mask].tolist()

    return fwd_indices, bwd_indices, seq_lens, seq_batch_indices

def SparseWindowedScaledDotProductSelfAttention(
    qkv: SparseTensor,
    window_size: int,
    shift_window: Tuple[int, int, int] = (0, 0, 0)
) -> SparseTensor:
    """
    Apply windowed scaled dot product self attention to a sparse tensor.

    Args:
        qkv (SparseTensor): [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
        window_size (int): The window size to use.
        shift_window (Tuple[int, int, int]): The shift of serialized coordinates.
        shift (int): The shift to use.
    """

    assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"

    serialization_spatial_cache_name = f'window_partition_{window_size}_{shift_window}'
    serialization_spatial_cache = qkv.GetSpatialCache(serialization_spatial_cache_name)
    if serialization_spatial_cache is None:
        fwd_indices, bwd_indices, seq_lens, seq_batch_indices = CalcWindowPartition(qkv, window_size, shift_window)
        qkv.RegisterSpatialCache(serialization_spatial_cache_name, (fwd_indices, bwd_indices, seq_lens, seq_batch_indices))
    else:
        fwd_indices, bwd_indices, seq_lens, seq_batch_indices = serialization_spatial_cache

    height = qkv.feats.shape[2]
    channels = qkv.feats.shape[3]
    
    qkv_feats = qkv.feats[fwd_indices]      # [M, 3, H, C]

    if all([seq_len == window_size for seq_len in seq_lens]):
        batch = len(seq_lens)
        num = window_size
        qkv_feats = qkv_feats.reshape(batch, num, 3, height, channels)
        q, k, v = qkv_feats.unbind(dim=2)                               # [B, N, H, C]
        out = MemEfficientAttention(q, k, v)                            # [B, N, H, C]
        out = out.reshape(batch * num, height, channels)                # [M, H, C]
    else:
        q, k, v = qkv_feats.unbind(dim=1)                               # [M, H, C]
        q = q.unsqueeze(0)                                              # [1, M, H, C]
        k = k.unsqueeze(0)                                              # [1, M, H, C]
        v = v.unsqueeze(0)                                              # [1, M, H, C]
        mask = BlockDiagonalMaskFromSeqlens(seq_lens)
        out = MemEfficientAttention(q, k, v, mask)[0]                   # [M, H, C]

    out = out[bwd_indices]      # [T, H, C]

    return qkv.replace(out)
