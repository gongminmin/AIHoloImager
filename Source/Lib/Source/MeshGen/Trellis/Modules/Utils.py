# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/modules/utils.py

from typing import *

import torch
import torch.nn as nn

from ..Modules import Sparse as sp

def ConvertModuleToFp16(l):
    """
    Convert primitive modules to float16.
    """

    fp16_modules = (
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.Linear,
        sp.SparseConv3D,
        sp.SparseInverseConv3D,
        sp.SparseLinear,
    )

    if isinstance(l, fp16_modules):
        for p in l.parameters():
            p.data = p.data.half()

def ZeroModule(module):
    """
    Zero out the parameters of a module and return it.
    """

    for p in module.parameters():
        p.detach().zero_()
    return module

def MemEfficientAttention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Memory-efficient attention using PyTorch's built-in scaled dot-product attention.
    """

    query = query.permute(0, 2, 1, 3)   # [N, H, L, C]
    key = key.permute(0, 2, 1, 3)   # [N, H, L, C]
    value = value.permute(0, 2, 1, 3)   # [N, H, L, C]
    if attn_mask != None:
        attn_mask = attn_mask.to(query.device)
    output = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask)
    return output.permute(0, 2, 1, 3)   # [N, L, H, C]

def BlockDiagonalMaskFromSeqlens(q_seqlen: Sequence[int], kv_seqlen: Optional[Sequence[int]] = None, causal = False) -> torch.Tensor:
    """
    Create a block-diagonal attention mask using nested tensors.

    Args:
        q_seqlen (Sequence[int]): Query sequence lengths.
        kv_seqlen (Optional[Sequence[int]]): Key/value sequence lengths.
        causal (bool): If True, apply causal masking within each sequence.

    Returns:
        torch.Tensor: Mask of shape (sum(q_seqlen), sum(q_seqlen)).
    """

    import numpy
    total_q_len = numpy.sum(q_seqlen)

    if kv_seqlen == None:
        kv_seqlen = q_seqlen
        total_kv_len = total_q_len
    else:
        assert len(q_seqlen) == len(kv_seqlen), "q_seqlen and kv_seqlen must have the same batch size"
        total_kv_len = numpy.sum(kv_seqlen)

    mask = torch.zeros(total_q_len, total_kv_len, dtype = torch.bool)

    q_start = 0
    kv_start = 0
    for q_len, kv_len in zip(q_seqlen, kv_seqlen):
        q_end = q_start + q_len
        kv_end = kv_start + kv_len
        block = torch.ones(q_len, kv_len, dtype = torch.bool)
        if causal:
            block = torch.tril(block[:, : min(q_len, kv_len)])
        mask[q_start : q_end, kv_start : kv_end] = block
        q_start = q_end
        kv_start = kv_end

    return mask.unsqueeze(0)
