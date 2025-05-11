# Copyright (c) 2025 Minmin Gong
#

# Simplified from https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/conv2d_layers.py

""" Conv2D w/ SAME padding, CondConv, MixedConv

A collection of conv layers and padding helpers needed by EfficientNet, MixNet, and
MobileNetV3 models that maintain weight compatibility with original Tensorflow models.

Copyright 2020 Ross Wightman
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn

def IsStaticPad(kernel_size, stride = 1, dilation = 1, **kwargs):
    return (stride == 1) and ((dilation * (kernel_size - 1)) % 2 == 0)

def GetPadding(kernel_size, stride = 1, dilation = 1, **kwargs):
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2

def CalcSamePad(i : int, k : int, s : int, d : int):
    return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)

def Conv2dSameFunc(x, weight : torch.Tensor, bias : Optional[torch.Tensor] = None, stride : Tuple[int, int] = (1, 1),
                   padding : Tuple[int, int] = (0, 0), dilation : Tuple[int, int] = (1, 1), groups : int = 1):
    ih, iw = x.size()[-2 : ]
    kh, kw = weight.size()[-2 : ]
    pad_h = CalcSamePad(ih, kh, stride[0], dilation[0])
    pad_w = CalcSamePad(iw, kw, stride[1], dilation[1])
    x = nn.functional.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return nn.functional.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)

class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1,
                 padding = 0, dilation = 1, groups = 1, bias = True, device : Optional[torch.device] = None):
        super(Conv2dSame, self).__init__(
              in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, device = device)

    def forward(self, x):
        return Conv2dSameFunc(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def GetPaddingValue(padding, kernel_size, **kwargs):
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == "same":
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if IsStaticPad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = GetPadding(kernel_size, **kwargs)
            else:
                # dynamic padding
                padding = 0
                dynamic = True
        elif padding == "valid":
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = GetPadding(kernel_size, **kwargs)
    return padding, dynamic

def CreateConv2dPad(in_channels, out_channels, kernel_size, **kwargs):
    padding = kwargs.pop("padding", "")
    kwargs.setdefault("bias", False)
    padding, is_dynamic = GetPaddingValue(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_channels, out_channels, kernel_size, **kwargs)
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding = padding, **kwargs)

def SelectConv2d(in_channels, out_channels, kernel_size, **kwargs):
    assert("groups" not in kwargs)  # only use 'depthwise' bool arg
    depthwise = kwargs.pop("depthwise", False)
    groups = out_channels if depthwise else 1
    return CreateConv2dPad(in_channels, out_channels, kernel_size, groups = groups, **kwargs)
