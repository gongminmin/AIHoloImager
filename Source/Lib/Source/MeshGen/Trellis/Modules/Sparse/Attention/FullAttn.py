# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/modules/sparse/attention/full_attn.py

from typing import *

import torch

from .. import SparseTensor
from ...Utils import MemEfficientAttention, BlockDiagonalMaskFromSeqlens

__all__ = [
    "SparseScaledDotProductAttention",
]

@overload
def SparseScaledDotProductAttention(qkv: SparseTensor) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        qkv (SparseTensor): A [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
    """
    ...

@overload
def SparseScaledDotProductAttention(q: SparseTensor, kv: Union[SparseTensor, torch.Tensor]) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, C] sparse tensor containing Qs.
        kv (SparseTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor or a [N, L, 2, H, C] dense tensor containing Ks and Vs.
    """
    ...

@overload
def SparseScaledDotProductAttention(q: torch.Tensor, kv: SparseTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, L, H, C] dense tensor containing Qs.
        kv (SparseTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor containing Ks and Vs.
    """
    ...

@overload
def SparseScaledDotProductAttention(q: SparseTensor, k: SparseTensor, v: SparseTensor) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (SparseTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (SparseTensor): A [N, *, H, Co] sparse tensor containing Vs.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...

@overload
def SparseScaledDotProductAttention(q: SparseTensor, k: torch.Tensor, v: torch.Tensor) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] dense tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] dense tensor containing Vs.
    """
    ...

@overload
def SparseScaledDotProductAttention(q: torch.Tensor, k: SparseTensor, v: SparseTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] dense tensor containing Qs.
        k (SparseTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (SparseTensor): A [N, *, H, Co] sparse tensor containing Vs.
    """
    ...

def SparseScaledDotProductAttention(*args, **kwargs):
    arg_names_dict = {
        1: ["qkv"],
        2: ["q", "kv"],
        3: ["q", "k", "v"]
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args) :]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs["qkv"]
        assert isinstance(qkv, SparseTensor), f"qkv must be a SparseTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"

        s = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
        kv_seqlen = q_seqlen
        qkv = qkv.feats     # [T, 3, H, C]
        q, k, v = qkv.unbind(dim = 1)
    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs["q"]
        kv = args[1] if len(args) > 1 else kwargs["kv"]
        assert isinstance(q, SparseTensor) and isinstance(kv, (SparseTensor, torch.Tensor)) or \
               isinstance(q, torch.Tensor) and isinstance(kv, SparseTensor), \
               f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"

        if isinstance(q, SparseTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, C]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            s = None
            num, length, height, channels = q.shape
            q_seqlen = [length] * num
            q = q.reshape(num * length, height, channels)   # [T_Q, H, C]

        if isinstance(kv, SparseTensor):
            assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [num, *, 2, H, C]"
            kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
            kv = kv.feats     # [T_KV, 2, H, C]
        else:
            assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            num, length, _, height, channels = kv.shape
            kv_seqlen = [length] * num
            kv = kv.reshape(num * length, 2, height, channels)   # [T_KV, 2, H, C]
        k, v = kv.unbind(dim = 1)
    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
        v = args[2] if len(args) > 2 else kwargs["v"]
        assert isinstance(q, SparseTensor) and isinstance(k, (SparseTensor, torch.Tensor)) and type(k) == type(v) or \
               isinstance(q, torch.Tensor) and isinstance(k, SparseTensor) and isinstance(v, SparseTensor), \
               f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"

        if isinstance(q, SparseTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, Ci]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            s = None
            num, length, height, channels_in = q.shape
            q_seqlen = [length] * num
            q = q.reshape(num * length, height, channels_in)  # [T_Q, H, Ci]

        if isinstance(k, SparseTensor):
            assert len(k.shape) == 3, f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert len(v.shape) == 3, f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            k = k.feats     # [T_KV, H, Ci]
            v = v.feats     # [T_KV, H, Co]
        else:
            assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
            assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
            num, length, height, channels_in, channels_out = *k.shape, v.shape[-1]
            kv_seqlen = [length] * num
            k = k.reshape(num * length, height, channels_in)     # [T_KV, H, Ci]
            v = v.reshape(num * length, height, channels_out)    # [T_KV, H, Co]

    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)
    mask = BlockDiagonalMaskFromSeqlens(q_seqlen, kv_seqlen)
    out = MemEfficientAttention(q, k, v, mask)[0]

    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(num, length, height, -1)
