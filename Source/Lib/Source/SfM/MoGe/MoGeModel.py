# Copyright (c) 2025 Minmin Gong
#

# Based on MoGe, https://github.com/microsoft/MoGe/blob/main/moge/model/v1.py

import importlib
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.utils import skip_init

from PythonSystem import GeneralDevice, WrapDinov2AttentionWithSdpa
from .Geometry import NormalizedViewPlaneUv, RecoverFocal

class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels : int = None,
        hidden_channels : int = None,
        padding_mode : str = "replicate",
        activation : Literal["relu", "leaky_relu", "silu", "elu"] = "relu",
        norm: Literal["group_norm", "layer_norm"] = "group_norm",
        device : Optional[torch.device] = None,
    ):
        super(ResidualConvBlock, self).__init__()

        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        if activation == "relu":
            activation_cls = lambda: nn.ReLU(inplace = True)
        elif activation == "leaky_relu":
            activation_cls = lambda: nn.LeakyReLU(negative_slope = 0.2, inplace = True)
        elif activation == "silu":
            activation_cls = lambda: nn.SiLU(inplace = True)
        elif activation == "elu":
            activation_cls = lambda: nn.ELU(inplace = True)
        else:
            raise ValueError(f'Unsupported activation function: {activation}')

        self.layers = nn.Sequential(
            nn.GroupNorm(1, in_channels, device = device),
            activation_cls(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size = 3, padding = 1, padding_mode = padding_mode, device = device),
            nn.GroupNorm(hidden_channels // 32 if norm == "group_norm" else 1, hidden_channels, device = device),
            activation_cls(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size = 3, padding = 1, padding_mode = padding_mode, device = device),
        )

        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size = 1, padding = 0, device = device) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        skip = self.skip_connection(x)
        x = self.layers(x)
        x = x + skip
        return x

class Head(nn.Module):
    def __init__(
        self,
        num_features : int,
        dim_in : int,
        dim_out : List[int],
        dim_proj : int = 512,
        dim_upsample : List[int] = [256, 128, 128],
        dim_times_res_block_hidden : int = 1,
        num_res_blocks : int = 1,
        res_block_norm : Literal["group_norm", "layer_norm"] = "group_norm",
        last_res_blocks : int = 0,
        last_conv_channels : int = 32,
        last_conv_size : int = 1,
        device : Optional[torch.device] = None,
    ):
        super().__init__()

        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels = dim_in, out_channels = dim_proj, kernel_size = 1, stride=1, padding = 0, device = device) for _ in range(num_features)
        ])

        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                self.MakeUpsampler(in_ch + 2, out_ch, device = device),
                *(ResidualConvBlock(out_ch, out_ch, dim_times_res_block_hidden * out_ch, activation = "relu", norm = res_block_norm, device = device) for _ in range(num_res_blocks))
            ) for in_ch, out_ch in zip([dim_proj] + dim_upsample[:-1], dim_upsample)
        ])

        self.output_block = nn.ModuleList([
            self.MakeOutputBlock(
                dim_upsample[-1] + 2, do, dim_times_res_block_hidden, last_res_blocks, last_conv_channels, last_conv_size, res_block_norm, device = device
            ) for do in dim_out
        ])

    def MakeUpsampler(self, in_channels : int, out_channels : int, device : Optional[torch.device] = None):
        upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2, device = device),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, padding_mode = "replicate", device = device)
        )
        upsampler[0].weight.data[ : ] = upsampler[0].weight.data[ : ,  : , : 1, : 1]
        return upsampler

    def MakeOutputBlock(
        self,
        dim_in : int,
        dim_out : int,
        dim_times_res_block_hidden : int,
        last_res_blocks : int,
        last_conv_channels: int,
        last_conv_size: int,
        res_block_norm: Literal["group_norm", "layer_norm"],
        device : Optional[torch.device] = None
    ):
        return nn.Sequential(
            nn.Conv2d(dim_in, last_conv_channels, kernel_size = 3, stride = 1, padding = 1, padding_mode = "replicate", device = device),
            *(ResidualConvBlock(last_conv_channels, last_conv_channels, dim_times_res_block_hidden * last_conv_channels, activation = "relu", norm = res_block_norm, device = device) for _ in range(last_res_blocks)),
            nn.ReLU(inplace = True),
            nn.Conv2d(last_conv_channels, dim_out, kernel_size = last_conv_size, stride = 1, padding = last_conv_size // 2, padding_mode = "replicate", device = device),
        )

    def forward(self, hidden_states : torch.Tensor, image : torch.Tensor):
        img_h, img_w = image.shape[-2 : ]
        patch_h, patch_w = img_h // 14, img_w // 14

        # Process the hidden states
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous())
                for proj, (feat, clstoken) in zip(self.projects, hidden_states)
        ], dim = 1).sum(dim = 1)

        # Upsample stage
        # (patch_h, patch_w) -> (patch_h * 2, patch_w * 2) -> (patch_h * 4, patch_w * 4) -> (patch_h * 8, patch_w * 8)
        for i, block in enumerate(self.upsample_blocks):
            # UV coordinates is for awareness of image aspect ratio
            uv = NormalizedViewPlaneUv(width = x.shape[-1], height = x.shape[-2], aspect_ratio = img_w / img_h, dtype = x.dtype, device = x.device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, uv], dim = 1)
            for layer in block:
                x = layer(x)

        # (patch_h * 8, patch_w * 8) -> (img_h, img_w)
        x = functional.interpolate(x, (img_h, img_w), mode = "bilinear", align_corners = False)
        uv = NormalizedViewPlaneUv(width = x.shape[-1], height = x.shape[-2], aspect_ratio = img_w / img_h, dtype = x.dtype, device = x.device)
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, uv], dim = 1)

        return [block(x) for block in self.output_block]

class MoGeModel(nn.Module):
    image_mean : torch.Tensor
    image_std : torch.Tensor

    def __init__(self, 
        encoder : str = "dinov2_vitb14",
        intermediate_layers : Union[int, List[int]] = 4,
        dim_proj : int = 512,
        dim_upsample : List[int] = [256, 128, 128],
        dim_times_res_block_hidden : int = 1,
        num_res_blocks : int = 1,
        remap_output : Literal[False, True, "linear", "sinh", "exp", "sinh_exp"] = "linear",
        res_block_norm : Literal["group_norm", "layer_norm"] = "group_norm",
        num_tokens_range : Tuple[int, int] = [1200, 2500],
        last_res_blocks : int = 0,
        last_conv_channels : int = 32,
        last_conv_size : int = 1,
        mask_threshold : float = 0.5,
        device : Optional[torch.device] = None,
        **deprecated_kwargs
    ):
        super(MoGeModel, self).__init__()

        if deprecated_kwargs:
            # Process legacy arguments
            if "trained_area_range" in deprecated_kwargs:
                num_tokens_range = [deprecated_kwargs["trained_area_range"][0] // 14 ** 2, deprecated_kwargs["trained_area_range"][1] // 14 ** 2]
                del deprecated_kwargs["trained_area_range"]

        self.encoder = encoder
        self.remap_output = remap_output
        self.intermediate_layers = intermediate_layers
        self.num_tokens_range = num_tokens_range
        self.mask_threshold = mask_threshold

        backbones_module = getattr(importlib.import_module("dinov2.hub.backbones"), encoder)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.backbone = backbones_module(pretrained = False)
        dim_feature = self.backbone.blocks[0].attn.qkv.in_features

        self.head = Head(
            num_features = intermediate_layers if isinstance(intermediate_layers, int) else len(intermediate_layers),
            dim_in = dim_feature,
            dim_out = [3, 1],
            dim_proj = dim_proj,
            dim_upsample = dim_upsample,
            dim_times_res_block_hidden = dim_times_res_block_hidden,
            num_res_blocks = num_res_blocks,
            res_block_norm = res_block_norm,
            last_res_blocks = last_res_blocks,
            last_conv_channels = last_conv_channels,
            last_conv_size = last_conv_size,
            device = device,
        )

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        self.EnablePytorchNativeSdpa()

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

    def EnablePytorchNativeSdpa(self):
        for i in range(len(self.backbone.blocks)):
            self.backbone.blocks[i].attn = WrapDinov2AttentionWithSdpa(self.backbone.blocks[i].attn)

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

    def forward(self, image : torch.Tensor, num_tokens : int) -> [torch.Tensor, torch.Tensor]:
        original_height, original_width = image.shape[-2 : ]

        # Resize to expected resolution defined by num_tokens
        resize_factor = ((num_tokens * 14 ** 2) / (original_height * original_width)) ** 0.5
        resized_width, resized_height = int(original_width * resize_factor), int(original_height * resize_factor)
        image = functional.interpolate(image, (resized_height, resized_width), mode = "bicubic", align_corners = False, antialias = True)

        # Apply image transformation for DINOv2
        image = (image - self.image_mean) / self.image_std
        image_14 = functional.interpolate(image, (resized_height // 14 * 14, resized_width // 14 * 14), mode = "bilinear", align_corners = False, antialias = True)

        # Get intermediate layers from the backbone
        features = self.backbone.get_intermediate_layers(image_14, self.intermediate_layers, return_class_token = True)

        # Predict points (and mask)
        points, mask = self.head(features, image)

        # Make sure fp32 precision for output
        with torch.autocast(device_type = image.device.type, dtype = torch.float32):
            # Resize to original resolution
            points = functional.interpolate(points, (original_height, original_width), mode = "bilinear", align_corners = False, antialias = False)
            mask = functional.interpolate(mask, (original_height, original_width), mode = "bilinear", align_corners = False, antialias = False)

            # Post-process points and mask
            points = points.permute(0, 2, 3, 1)
            points = self.RemapPoints(points)     # slightly improves the performance in case of very large output values
            mask = mask.squeeze(1)

        return points, mask

    @torch.inference_mode()
    def Focal(
        self,
        image: torch.Tensor,
        resolution_level: int = 9,
        num_tokens: int = None,
        use_fp16: bool = True,
    ) -> Dict[str, torch.Tensor]:
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

        original_height, original_width = image.shape[-2 :]
        area = original_height * original_width
        aspect_ratio = original_width / original_height

        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        with torch.autocast(device_type = image.device.type, dtype = torch.float16, enabled = use_fp16):
            points, mask = self.forward(image, num_tokens)

        mask_binary = mask > self.mask_threshold

        # Get camera-space point map. (Focal here is the focal length relative to half the image diagonal)
        focal = RecoverFocal(points, mask_binary)
        fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
        return fy.item() * original_height
