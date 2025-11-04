# Copyright (c) 2025 Minmin Gong
#

# Based on MoGe, https://github.com/microsoft/MoGe/blob/main/moge/utils/geometry_numpy.py and https://github.com/microsoft/MoGe/blob/main/moge/utils/geometry_torch.py

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as functional

@torch.enable_grad()
def SolveOptimalFocalShift(uv : torch.Tensor, xyz : torch.Tensor, focal: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal"

    criterion = nn.MSELoss(reduction = "sum")
    shift = nn.Parameter(torch.zeros(1, dtype = xyz.dtype, device = xyz.device))

    lr_base = 1.0
    max_iter = 10

    optimizer = torch.optim.LBFGS([shift], lr = lr_base, max_iter = max_iter, line_search_fn = "strong_wolfe", tolerance_grad = 1e-3, tolerance_change = 1e-3)

    def Closure():
        optimizer.zero_grad()

        xy_proj   = xyz[..., :2] / (xyz[..., 2] + shift).unsqueeze(-1)

        if focal is None:
            f = (xy_proj * uv).sum() / torch.square(xy_proj).sum()
        else:
            f = focal

        pred_uv = f * xy_proj

        loss = criterion(pred_uv, uv)

        loss.backward()
        return loss

    prev_loss = 1e10
    for it in range(max_iter):
        loss = optimizer.step(Closure)

        loss_val = loss.item()
        if abs(prev_loss - loss_val) < 1e-3 * max(1.0, loss_val):
            print(f"Converged at iteration {it}, loss {loss_val:.7f}")
            break
        prev_loss = loss_val

        if it % 5 == 0:
            print(f"Iteration {it}, loss {loss_val:.7f}")

    optim_shift = shift[0].detach()

    if focal is None:
        xy_proj = xyz[..., : 2] / (xyz[..., 2] + optim_shift).unsqueeze(-1)
        optim_focal = ((xy_proj * uv).sum() / torch.square(xy_proj).sum()).detach()
    else:
        optim_focal = focal

    return optim_focal, optim_shift

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

def RecoverFocalShift(points: torch.Tensor, mask: Optional[torch.Tensor] = None, focal: Optional[torch.Tensor] = None, downsample_size: Optional[Tuple[int, int]] = (64, 64)):
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Note that it assumes:
    - the optical center is at the center of the map
    - the map is undistorted
    - the map is isometric in the x and y directions

    ### Parameters:
    - `points: torch.Tensor` of shape (..., H, W, 3)
    - `mask: torch.Tensor` of shape (..., H, W). Optional.
    - `focal: torch.Tensor` of shape (...). Optional.
    - `downsample_size: Tuple[int, int]` in (height, width), the size of the downsampled map. Downsampling produces approximate solution and is efficient for large maps.

    ### Returns:
    - `focal`: torch.Tensor of shape (...) the estimated focal length, relative to the half diagonal of the map
    - `shift`: torch.Tensor of shape (...) Z-axis shift to translate the point map to camera space
    """
    shape = points.shape
    width = shape[-2]
    height = shape[-3]
    diagonal = (height ** 2 + width ** 2) ** 0.5

    points = points.reshape(-1, *shape[-3 :])
    mask = None if mask is None else mask.reshape(-1, *shape[-3 : -1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = NormalizedViewPlaneUv(width, height, dtype = points.dtype, device = points.device)  # (H, W, 2)

    points_lr = functional.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode = "nearest").permute(0, 2, 3, 1)
    uv_lr = functional.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode = "nearest").squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else functional.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode = "nearest").squeeze(1) > 0

    points_lr = points_lr.detach()
    optim_focal = []
    optim_shift = []
    for i in range(points.shape[0]):
        points_lr_i = points_lr[i] if mask is None else points_lr[i][mask_lr[i]]
        uv_lr_i = uv_lr if mask is None else uv_lr[mask_lr[i]]
        optim_focal_i, optim_shift_i = SolveOptimalFocalShift(uv_lr_i, points_lr_i, None if focal is None else focal[i])
        if focal is None:
            optim_focal.append(float(optim_focal_i))
        optim_shift.append(float(optim_shift_i))
    optim_shift = torch.tensor(optim_shift, device = points.device, dtype = points.dtype).reshape(shape[: -3])

    if focal is None:
        optim_focal = torch.tensor(optim_focal, device = points.device, dtype = points.dtype).reshape(shape[: -3])
    else:
        optim_focal = focal.reshape(shape[: -3])

    return optim_focal, optim_shift

def IntrinsicsFromFocalCenter(
    fx: torch.Tensor,
    fy: torch.Tensor,
    cx: Union[float, torch.Tensor],
    cy: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Get OpenCV intrinsics matrix

    Args:
        focal_x (torch.Tensor): focal length in x axis
        focal_y (torch.Tensor): focal length in y axis
        cx (float | torch.Tensor): principal point in x axis
        cy (float | torch.Tensor): principal point in y axis

    Returns:
        (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    if isinstance(cx, float):
        cx = torch.tensor([cx], dtype = fx.dtype, device = fx.device)
    if isinstance(cy, float):
        cy = torch.tensor([cy], dtype = fx.dtype, device = fx.device)
    N = fx.shape[0]
    zeros = torch.zeros(N, dtype = fx.dtype, device = fx.device)
    ones = torch.ones(N, dtype = fx.dtype, device = fx.device)
    ret = torch.stack([fx, zeros, cx, zeros, fy, cy, zeros, zeros, ones], dim = -1)
    ret = ret.unflatten(-1, (3, 3))
    return ret

def UnprojectCV(
    uv_coord: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Unproject uv coordinates to 3D view space following the OpenCV convention

    Args:
        uv_coord (torch.Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        depth (torch.Tensor): [..., N] depth value
        intrinsics (torch.Tensor): [..., 3, 3] intrinsics matrix.
        extrinsics (torch.Tensor): [..., 4, 4] extrinsics matrix. Optional.

    Returns:
        points (torch.Tensor): [..., N, 3] 3d points
    """

    points = torch.cat([uv_coord, torch.ones_like(uv_coord[..., : 1])], dim = -1)
    points = points @ torch.inverse(intrinsics).transpose(-2, -1)
    points = points * depth[..., None]
    if extrinsics is not None:
        points = torch.cat([points, torch.ones_like(points[..., : 1])], dim = -1)
        points = (points @ torch.inverse(extrinsics).transpose(-2, -1))[..., : 3]
    return points

def ImageUV(height: int, width: int, left: Optional[int] = None, top: Optional[int] = None, right: Optional[int] = None, bottom: Optional[int] = None, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Get image space UV grid, ranging in [0, 1]. 

    >>> image_uv(10, 10):
    [[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
     [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
      ...             ...                  ...
     [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]

    Args:
        width (int): image width
        height (int): image height

    Returns:
        torch.Tensor: shape (height, width, 2)
    """

    if left is None:
        left = 0
    if top is None:
        top = 0
    if right is None:
        right = width
    if bottom is None:
        bottom = height
    u = torch.linspace((left + 0.5) / width, (right - 0.5) / width, right - left, device = device, dtype = dtype)
    v = torch.linspace((top + 0.5) / height, (bottom - 0.5) / height, bottom - top, device = device, dtype = dtype)
    u, v = torch.meshgrid(u, v, indexing = "xy")
    uv = torch.stack([u, v], dim = -1)
    return uv
