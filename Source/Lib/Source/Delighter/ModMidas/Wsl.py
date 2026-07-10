# Copyright (c) 2025-2026 Minmin Gong
#

# Inspired by https://github.com/facebookresearch/WSL-Images/blob/main/hubconf.py

import torch
from torchvision.models import resnet

def Resnext101_32(width_per_group: int) -> torch.Tensor:
    kwargs = {}
    kwargs["groups"] = 32
    kwargs["width_per_group"] = width_per_group
    return resnet.ResNet(resnet.Bottleneck, [3, 4, 23, 3], **kwargs)
