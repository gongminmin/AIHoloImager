# Copyright (c) 2025 Minmin Gong
#

# Based on MoGe, https://github.com/microsoft/MoGe/blob/main/moge/utils/geometry_numpy.py and https://github.com/microsoft/MoGe/blob/main/moge/utils/geometry_torch.py

from functools import partial
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

def SolveOptimalFocal(uv : np.ndarray, xyz : np.ndarray):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal"
    uv = uv.reshape(-1, 2)
    xy = xyz[..., : 2].reshape(-1, 2)
    z = xyz[..., 2].reshape(-1)

    def Func(uv : np.ndarray, xy : np.ndarray, z : np.ndarray, shift : np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    from scipy.optimize import least_squares
    solution = least_squares(partial(Func, uv, xy, z), x0 = 0, ftol = 1e-3, method = "lm")
    optim_shift = solution["x"].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[:, None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_focal

def NormalizedViewPlaneUv(width : int, height : int, aspect_ratio : Optional[float] = None, dtype : Optional[torch.dtype] = None, device : Optional[torch.device] = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5
    span_x = aspect_ratio * span_y

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype = dtype, device = device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype = dtype, device = device)
    u, v = torch.meshgrid(u, v, indexing = "xy")
    uv = torch.stack([u, v], dim = -1)
    return uv

def RecoverFocal(points : torch.Tensor, mask : Optional[torch.Tensor] = None, downsample_size: Optional[Tuple[int, int]] = (64, 64)):
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Note that it assumes:
    - the optical center is at the center of the map
    - the map is undistorted
    - the map is isometric in the x and y directions

    ### Parameters:
    - `points: torch.Tensor` of shape (..., H, W, 3)
    - `downsample_size: Tuple[int, int]` in (height, width), the size of the downsampled map. Downsampling produces approximate solution and is efficient for large maps.

    ### Returns:
    - `focal`: torch.Tensor of shape (...) the estimated focal length, relative to the half diagonal of the map
    """
    shape = points.shape
    width = shape[-2]
    height = shape[-3]

    points = points.reshape(-1, *shape[-3 : ])
    mask = None if mask is None else mask.reshape(-1, *shape[-3 : -1])
    uv = NormalizedViewPlaneUv(width, height, dtype = points.dtype, device = points.device)  # (H, W, 2)

    points_lr = functional.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode = "nearest").permute(0, 2, 3, 1)
    uv_lr = functional.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode = "nearest").squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else functional.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode = "nearest").squeeze(1) > 0
    
    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
    optim_focal = []
    for i in range(points.shape[0]):
        points_lr_i_np = points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]
        if uv_lr_i_np.shape[0] < 2:
            optim_focal.append(1)
            continue
        optim_focal_i = SolveOptimalFocal(uv_lr_i_np, points_lr_i_np)
        optim_focal.append(float(optim_focal_i))

    optim_focal = torch.tensor(optim_focal, dtype = points.dtype, device = points.device).reshape(shape[ : -3])
    return optim_focal
