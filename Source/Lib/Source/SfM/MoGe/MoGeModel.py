# Copyright (c) 2025-2026 Minmin Gong
#

# Based on MoGe 2, https://github.com/microsoft/MoGe/blob/main/moge/model/v2.py

import importlib
from numbers import Number
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.utils import skip_init

from PythonSystem import GeneralDevice
from .Geometry import NormalizedViewPlaneUv, RecoverFocalShift, IntrinsicsFromFocalCenter, UnprojectCV, ImageUV
from .Modules import Dinov2Encoder, Mlp, ConvStack

class MoGeModel(nn.Module):
    encoder: Dinov2Encoder
    neck: ConvStack
    points_head: ConvStack
    mask_head: ConvStack
    scale_head: Mlp

    def __init__(self, 
        encoder: Dict[str, Any],
        neck: Dict[str, Any],
        points_head: Dict[str, Any],
        mask_head: Dict[str, Any] = None,
        scale_head: Dict[str, Any] = None,
        remap_output: Literal["linear", "sinh", "exp", "sinh_exp"] = "linear",
        num_tokens_range: List[int] = [1200, 3600],
        device : Optional[torch.device] = None,
        **deprecated_kwargs
    ):
        super(MoGeModel, self).__init__()

        if deprecated_kwargs:
            warnings.warn(f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}")

        self.remap_output = remap_output
        self.num_tokens_range = num_tokens_range

        self.encoder = Dinov2Encoder(**encoder, device = device)
        self.neck = ConvStack(**neck, device = device)
        self.points_head = ConvStack(**points_head, device = device)
        if mask_head is not None:
            self.mask_head = ConvStack(**mask_head, device = device)
        else:
            self.mask_head = None
        if scale_head is not None:
            self.scale_head = Mlp(**scale_head, device = device)
        else:
            self.scale_head = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @classmethod
    def FromPretrained(cls, model_path : Union[str, Path], model_kwargs : Optional[Dict[str, Any]] = None) -> "MoGeModel":
        """
        Load a model from a checkpoint file.

        ### Parameters:
        - `model_path`: path to the checkpoint file.
        - `model_kwargs`: additional keyword arguments to override the parameters in the checkpoint.

        ### Returns:
        - A new instance of `MoGe` with the parameters loaded from the checkpoint.
        """

        checkpoint = torch.load(model_path, map_location = GeneralDevice(), weights_only = True)

        model_config = checkpoint["model_config"]
        if model_kwargs is not None:
            model_config.update(model_kwargs)
        model = skip_init(cls, **model_config)
        model.load_state_dict(checkpoint["model"])

        return model

    def RemapPoints(self, points : torch.Tensor) -> torch.Tensor:
        if self.remap_output == "linear":
            pass
        elif self.remap_output =="sinh":
            points = torch.sinh(points)
        elif self.remap_output == "exp":
            xy, z = points.split([2, 1], dim = -1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim = -1)
        elif self.remap_output == "sinh_exp":
            xy, z = points.split([2, 1], dim = -1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim = -1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")
        return points

    def forward(self, image : torch.Tensor, num_tokens: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _, img_h, img_w = image.shape
        device, dtype = image.device, image.dtype

        aspect_ratio = img_w / img_h
        base_w = round((num_tokens * aspect_ratio) ** 0.5)
        base_h = round((num_tokens / aspect_ratio) ** 0.5)

        # Backbones encoding
        features, cls_token = self.encoder(image, base_h, base_w, return_class_token = True)
        features = [features, None, None, None, None]

        # Concat UVs for aspect ratio input
        for level in range(5):
            uv = NormalizedViewPlaneUv(width = base_w * 2 ** level, height = base_h * 2 ** level, aspect_ratio = aspect_ratio, dtype = dtype, device = device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat((features[level], uv), dim = 1)

        # Shared neck
        features = self.neck(features)

        # Heads decoding
        points = self.points_head(features)[-1]
        # Resize
        points = functional.interpolate(points, (img_h, img_w), mode = "bilinear", align_corners = False, antialias = False)
        # Remap output
        points = points.permute(0, 2, 3, 1)
        points = self.RemapPoints(points)     # slightly improves the performance in case of very large output values

        if self.mask_head is not None:
            mask = self.mask_head(features)[-1]
            mask = functional.interpolate(mask, (img_h, img_w), mode = "bilinear", align_corners = False, antialias = False)
            mask = mask.squeeze(1).sigmoid()
        else:
            mask = None

        if self.scale_head is not None:
            metric_scale = self.scale_head(cls_token)
            metric_scale = metric_scale.squeeze(1).exp()
        else:
            metric_scale = None

        return points, mask, metric_scale

    @torch.no_grad()
    def Focal(
        self,
        image: torch.Tensor,
        resolution_level: int = 9,
        num_tokens: int = None,
        use_fp16: bool = True,
    ) -> float:
        """
        User-friendly inference function

        ### Parameters
        - `image`: input image tensor of shape (B, 3, H, W) or (3, H, W)
        - `resolution_level`: the resolution level to use for the output point map in 0-9. Default: 9 (highest)
            
        ### Returns

        focal length in pixels.
        """

        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(dtype = self.dtype, device = self.device)

        original_height, original_width = image.shape[-2 :]
        area = original_height * original_width
        aspect_ratio = original_width / original_height

        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        with torch.autocast(device_type = self.device.type, dtype = torch.float16, enabled = (use_fp16 and self.dtype != torch.float16)):
            points, mask, _ = self.forward(image, num_tokens = num_tokens)

        # Always process the output in fp32 precision
        points = points.to(torch.float32)

        if mask is not None:
            mask_binary = mask > 0.5
        else:
            mask_binary = None

        # Convert affine point map to camera-space. Recover depth and intrinsics from point map.
        # NOTE: Focal here is the focal length relative to half the image diagonal
        # Recover focal and shift from predicted point map
        focal, _ = RecoverFocalShift(points, mask_binary)
        fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
        return fy.item() * original_height

    @torch.no_grad()
    def PointCloud(
        self, 
        image: torch.Tensor,
        fov_x: Union[Number, torch.Tensor],
        resolution_level: int = 9,
        num_tokens : int = None,
        use_fp16 : bool = True,
    ) -> torch.Tensor:
        """
        User-friendly inference function

        ### Parameters
        - `image`: input image tensor of shape (B, 3, H, W) or (3, H, W)
        - `resolution_level`: the resolution level to use for the output point map in 0-9. Default: 9 (highest)
        - `fov_x`: the horizontal camera FoV in degrees. If None, it will be inferred from the predicted point map. Default: None
            
        ### Returns
        point cloud in camera space, tensor of shape (B, H, W, 3) or (H, W, 3).
        """

        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False
        image = image.to(dtype = self.dtype, device = self.device)

        original_height, original_width = image.shape[-2 :]
        area = original_height * original_width
        aspect_ratio = original_width / original_height

        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        with torch.autocast(device_type = self.device.type, dtype = torch.float16, enabled = (use_fp16 and self.dtype != torch.float16)):
            points, mask, metric_scale = self.forward(image, num_tokens = num_tokens)

        # Always process the output in fp32 precision
        points = points.to(torch.float32)
        if isinstance(fov_x, torch.Tensor):
            fov_x = fov_x.to(torch.float32)
        else:
            fov_x = torch.tensor(fov_x, device = self.device, dtype = torch.float32)

        if mask is not None:
            mask_binary = mask > 0.5
        else:
            mask_binary = None

        # Convert affine point map to camera-space. Recover depth and intrinsics from point map.
        # NOTE: Focal here is the focal length relative to half the image diagonal
        # Focal is known, recover shift only
        focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(fov_x / 2))
        if focal.ndim == 0:
            focal = focal.unsqueeze(-1).expand(points.shape[0])
        _, shift = RecoverFocalShift(points, mask_binary, focal = focal)
        fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
        fx = fy / aspect_ratio
        intrinsics = IntrinsicsFromFocalCenter(fx, fy, 0.5, 0.5)
        depth = points[..., 2].clone() + shift

        if metric_scale is not None:
            depth *= metric_scale.to(torch.float32)

        # Recompute the point map using the actual depth map & intrinsics
        points = UnprojectCV(ImageUV(width = depth.shape[-1], height = depth.shape[-2], dtype = points.dtype, device = points.device), depth, intrinsics = intrinsics.unsqueeze(-3))

        if mask_binary is not None:
            mask_binary &= depth > 0        # in case depth is contains negative values (which should never happen in practice)
            points = torch.where(mask_binary.unsqueeze(-1), points, torch.inf)

        if omit_batch_dim:
            points = points.squeeze(0)

        return points
