# Copyright (c) 2025 Minmin Gong
#

# Simplified from https://github.com/CCareaga/MiDaS/blob/master/altered_midas/midas_net.py
# and https://github.com/CCareaga/MiDaS/blob/master/altered_midas/midas_net_custom.py

"""
MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""

from typing import Optional

import torch
import torch.nn as nn

from .Blocks import FeatureFusionBlock, Interpolate, MakeEncoder

class MidasNet(nn.Module):
    def __init__(self, activation = "sigmoid", features = 256, in_channels = 3, out_channels = 1, group_width = 8, last_residual = False, device : Optional[torch.device] = None):
        super(MidasNet, self).__init__()

        self.out_channels = out_channels
        self.last_residual = last_residual

        self.pretrained, self.scratch = MakeEncoder(
            backbone = "resnext101_wsl",
            features = features,
            in_channels = in_channels,
            group_width = group_width,
            device = device,
        )

        self.scratch.refinenet4 = FeatureFusionBlock(features, device = device)
        self.scratch.refinenet3 = FeatureFusionBlock(features, device = device)
        self.scratch.refinenet2 = FeatureFusionBlock(features, device = device)
        self.scratch.refinenet1 = FeatureFusionBlock(features, device = device)

        if activation == "sigmoid":
            out_act = nn.Sigmoid()
        elif activation == "relu":
            out_act = nn.ReLU()
        else:
            out_act = nn.Identity()

        res_dim = 128 + (in_channels if last_residual else 0)
        self.scratch.output_conv = nn.ModuleList([
            nn.Conv2d(features, 128, kernel_size = 3, stride = 1, padding = 1, device = device),
            Interpolate(scale_factor = 2, mode = "bilinear"),
            nn.Conv2d(res_dim, 32, kernel_size = 3, stride = 1, padding = 1, device = device),
            nn.ReLU(True),
            nn.Conv2d(32, out_channels, kernel_size = 1, stride = 1, padding = 0, device = device),
            out_act
        ])

    def forward(self, x):
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv[0](path_1)
        out = self.scratch.output_conv[1](out)

        if self.last_residual:
            out = torch.cat((out, x), dim = 1)

        out = self.scratch.output_conv[2](out)
        out = self.scratch.output_conv[3](out)
        out = self.scratch.output_conv[4](out)
        out = self.scratch.output_conv[5](out)

        return out

class MidasNetSmall(nn.Module):
    def __init__(self, activation = "sigmoid", features = 64, in_channels = 3, out_channels = 1, out_bias = 0, device : Optional[torch.device] = None):
        super(MidasNetSmall, self).__init__()

        self.pretrained, self.scratch = MakeEncoder(
            backbone = "efficientnet_lite3",
            features = features,
            in_channels = in_channels,
            expand = True,
            device = device,
        )
  
        self.scratch.activation = nn.ReLU(False)    

        self.scratch.refinenet4 = FeatureFusionBlock(features * 8, self.scratch.activation, expand = True, custom = True, device = device)
        self.scratch.refinenet3 = FeatureFusionBlock(features * 4, self.scratch.activation, expand = True, custom = True, device = device)
        self.scratch.refinenet2 = FeatureFusionBlock(features * 2, self.scratch.activation, expand = True, custom = True, device = device)
        self.scratch.refinenet1 = FeatureFusionBlock(features * 1, self.scratch.activation, expand = False, custom = True, device = device)

        if activation == "sigmoid":
            output_act = nn.Sigmoid()
        elif activation == "tanh":
            output_act = nn.Tanh()
        else:
            output_act = nn.Identity()
        
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size = 3, stride = 1, padding = 1, groups = 1, device = device),
            Interpolate(scale_factor = 2, mode = "bilinear"),
            nn.Conv2d(features // 2, 32, kernel_size = 3, stride = 1, padding = 1, device = device),
            self.scratch.activation,
            nn.Conv2d(32, out_channels, kernel_size = 1, stride = 1, padding = 0, device = device),
            output_act
        )
        self.scratch.output_conv[-2].bias = torch.nn.Parameter(torch.ones(out_channels, device = device) * out_bias)

    def forward(self, x):
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv(path_1)

        return out
