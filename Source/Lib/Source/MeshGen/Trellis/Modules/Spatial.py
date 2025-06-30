# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/modules/spatial.py

import torch

def PixelShuffle3D(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """
    3D pixel shuffle.
    """
    batch, channels, height, width, depth = x.shape
    channels = channels // (scale_factor ** 3)
    x = x.reshape(batch, channels, scale_factor, scale_factor, scale_factor, height, width, depth)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(batch, channels, height * scale_factor, width * scale_factor, depth * scale_factor)
    return x

def Patchify(x: torch.Tensor, patch_size: int):
    """
    Patchify a tensor.

    Args:
        x (torch.Tensor): (N, C, *spatial) tensor
        patch_size (int): Patch size
    """
    dim = x.dim() - 2
    for d in range(2, dim + 2):
        assert x.shape[d] % patch_size == 0, f"Dimension {d} of input tensor must be divisible by patch size, got {x.shape[d]} and {patch_size}"

    x = x.reshape(*x.shape[:2], *sum([[x.shape[d] // patch_size, patch_size] for d in range(2, dim + 2)], []))
    x = x.permute(0, 1, *([2 * i + 3 for i in range(dim)] + [2 * i + 2 for i in range(dim)]))
    x = x.reshape(x.shape[0], x.shape[1] * (patch_size ** dim), *(x.shape[-dim :]))
    return x

def Unpatchify(x: torch.Tensor, patch_size: int):
    """
    Unpatchify a tensor.

    Args:
        x (torch.Tensor): (N, C, *spatial) tensor
        patch_size (int): Patch size
    """
    dim = x.dim() - 2
    assert x.shape[1] % (patch_size ** dim) == 0, f"Second dimension of input tensor must be divisible by patch size to unpatchify, got {x.shape[1]} and {patch_size ** dim}"

    x = x.reshape(x.shape[0], x.shape[1] // (patch_size ** dim), *([patch_size] * dim), *(x.shape[-dim :]))
    x = x.permute(0, 1, *(sum([[2 + dim + i, 2 + i] for i in range(dim)], [])))
    x = x.reshape(x.shape[0], x.shape[1], *[x.shape[2 + 2 * i] * patch_size for i in range(dim)])
    return x
