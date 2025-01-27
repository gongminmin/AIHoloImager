# Copyright (c) 2025 Minmin Gong
#

# Simplified from https://github.com/CCareaga/MiDaS/blob/master/altered_midas/blocks.py

import torch
import torch.nn as nn

from .Effnet.Conv2dLayers import Conv2dSame
from .Effnet.EfficientNet import TfEfficientNetLite3
from .Wsl import Resnext101_32

def MakeEncoder(backbone, features, expand = False, in_channels = 3, group_width = 8):
    if backbone == "resnext101_wsl":
        model = MakeResnext101(in_channels, group_width)
        scratch = MakeScratch((256, 512, 1024, 2048), features, expand)
    elif backbone == "efficientnet_lite3":
        model = MakeEfficientNetLite3(in_channels)
        scratch = MakeScratch((32, 48, 136, 384), features, expand)
    else:
        print(f"Backbone '{backbone}' is not implemented")
        assert(False)

    return model, scratch

def MakeScratch(in_shape, out_shape, expand = False):
    scratch = nn.Module()

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8
    else:
        out_shape1 = out_shape
        out_shape2 = out_shape
        out_shape3 = out_shape
        if len(in_shape) >= 4:
            out_shape4 = out_shape

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size = 3, stride = 1, padding = 1, bias = False, groups = 1
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size = 3, stride = 1, padding = 1, bias = False, groups = 1
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size = 3, stride = 1, padding = 1, bias = False, groups = 1
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size = 3, stride = 1, padding = 1, bias = False, groups = 1
        )

    return scratch

def MakeResnext101(in_channels = 3, group_width = 8):
    resnet = Resnext101_32(group_width)
    if in_channels != 3:
        resnet.conv1 = torch.nn.Conv2d(in_channels, 64, 7, 2, 3, bias = False)
    
    model = nn.Module()
    model.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )
    model.layer2 = resnet.layer2
    model.layer3 = resnet.layer3
    model.layer4 = resnet.layer4

    return model

def MakeEfficientNetLite3(in_channels = 3):
    effnet = TfEfficientNetLite3()
    if in_channels != 3:
        effnet.conv_stem = Conv2dSame(in_channels, 32, kernel_size = (3, 3), stride = (2, 2), bias = False)

    model = nn.Module()
    model.layer1 = nn.Sequential(
        effnet.conv_stem, effnet.bn1, effnet.act1, *effnet.blocks[0 : 2]
    )
    model.layer2 = nn.Sequential(*effnet.blocks[2 : 3])
    model.layer3 = nn.Sequential(*effnet.blocks[3 : 5])
    model.layer4 = nn.Sequential(*effnet.blocks[5 : 9])

    return model

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners = False):
        super(Interpolate, self).__init__()

        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = nn.functional.interpolate(
            x, scale_factor = self.scale_factor, mode = self.mode, align_corners = self.align_corners
        )
        return x

class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation = nn.ReLU(True)):
        super(ResidualConvUnit, self).__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size = 3, stride = 1, padding = 1, bias = True
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size = 3, stride = 1, padding = 1, bias = True
        )

        self.activation = activation

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)

        out = self.activation(out)
        out = self.conv2(out)

        return out + x

class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation = nn.ReLU(True), expand = False, custom = False):
        super(FeatureFusionBlock, self).__init__()

        if custom:
            out_features = features
            if expand:
                out_features //= 2
            self.out_conv = nn.Conv2d(features, out_features, kernel_size = 1, stride = 1, padding = 0, bias = True, groups = 1)

        self.res_conv_unit1 = ResidualConvUnit(features, activation)
        self.res_conv_unit2 = ResidualConvUnit(features, activation)

    def forward(self, *xs, size = None):
        output = xs[0]

        if len(xs) == 2:
            output += self.res_conv_unit1(xs[1])

        output = self.res_conv_unit2(output)

        if (not hasattr(self, "out_conv")) or (size is None):
            modifier = {"scale_factor" : 2}
        else:
            modifier = {"size" : size}

        output = nn.functional.interpolate(output, **modifier, mode = "bilinear", align_corners = True)

        if hasattr(self, "out_conv"):
            output = self.out_conv(output)

        return output
