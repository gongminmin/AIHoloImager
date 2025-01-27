# Copyright (c) 2025 Minmin Gong
#

# Simplified from https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/gen_efficientnet.py
# and https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/efficientnet_builder.py

""" EfficientNet / MobileNetV3 Blocks and Builder

Copyright 2020 Ross Wightman
"""

from copy import deepcopy
import math

import torch.nn as nn

from .Conv2dLayers import SelectConv2d

# Defaults used for Google/Tensorflow training of mobile networks /w RMSprop as per
# papers and TF reference implementations. PT momentum equiv for TF decay is (1 - TF decay)
# NOTE: momentum varies btw .99 and .9997 depending on source
# .99 in official TF TPU impl
# .9997 (/w .999 in search space) for paper
#
# PyTorch defaults are momentum = .1, eps = 1e-5
#
BatchNormEpsTfDefault = 1e-3

def Sigmoid(x, inplace: bool = False):
    return x.sigmoid_() if inplace else x.sigmoid()

def MakeDivisible(v : int, divisor : int = 8, min_value : int = None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:  # ensure round down does not go down by more than 10%.
        new_v += divisor
    return new_v

def RoundChannels(channels, multiplier = 1.0, divisor = 8, channel_min = None):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels
    channels *= multiplier
    return MakeDivisible(channels, divisor, channel_min)

class BlockDefine:
    def __init__(self, block_type : str, kernel_size : int, stride : int, exp_ratio : float, out_channels : int):
        self.block_type = block_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.exp_ratio = exp_ratio
        self.out_channels = out_channels

class ArchDefine:
    def __init__(self, block_def : BlockDefine, num_repeat : int):
        self.block_def = block_def
        self.num_repeat = num_repeat

class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks with an expansion
    factor of 1.0. This is an alternative to having a IR with optional first pw conv.
    """
    def __init__(self, block_def : BlockDefine, in_channels, pad_type = "", act_layer = nn.ReLU,
                 norm_layer = nn.BatchNorm2d, norm_kwargs = None):
        super(DepthwiseSeparableConv, self).__init__()

        assert(block_def.stride in [1, 2])
        norm_kwargs = norm_kwargs or {}
        self.has_residual = (block_def.stride == 1) and (in_channels == block_def.out_channels)

        self.conv_dw = SelectConv2d(
            in_channels, in_channels, block_def.kernel_size, stride = block_def.stride, padding = pad_type, depthwise = True)
        self.bn1 = norm_layer(in_channels, **norm_kwargs)
        self.act1 = act_layer(inplace = True)

        self.se = nn.Identity()

        self.conv_pw = SelectConv2d(in_channels, block_def.out_channels, 1, padding = pad_type)
        self.bn2 = norm_layer(block_def.out_channels, **norm_kwargs)
        self.act2 = nn.Identity()

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.has_residual:
            x += residual
        return x

class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE"""

    def __init__(self, block_def : BlockDefine, in_channels, pad_type = "", act_layer = nn.ReLU,
                 norm_layer = nn.BatchNorm2d, norm_kwargs = None):
        super(InvertedResidual, self).__init__()

        norm_kwargs = norm_kwargs or {}
        mid_channels = int(MakeDivisible(in_channels * block_def.exp_ratio))
        self.has_residual = (in_channels == block_def.out_channels) and (block_def.stride == 1)

        # Point-wise expansion
        self.conv_pw = SelectConv2d(in_channels, mid_channels, 1, padding = pad_type)
        self.bn1 = norm_layer(mid_channels, **norm_kwargs)
        self.act1 = act_layer(inplace = True)

        # Depth-wise convolution
        self.conv_dw = SelectConv2d(
            mid_channels, mid_channels, block_def.kernel_size, stride = block_def.stride, padding = pad_type, depthwise = True)
        self.bn2 = norm_layer(mid_channels, **norm_kwargs)
        self.act2 = act_layer(inplace = True)

        self.se = nn.Identity()  # for jit.script compat

        # Point-wise linear projection
        self.conv_pwl = SelectConv2d(mid_channels, block_def.out_channels, 1, padding = pad_type)
        self.bn3 = norm_layer(block_def.out_channels, **norm_kwargs)

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            x += residual
        return x

class EfficientNetBuilder:
    """ Build Trunk Blocks for Efficient/Mobile Networks

    This ended up being somewhat of a cross between
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    and
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_builder.py
    """

    def __init__(self, channel_multiplier = 1.0, channel_divisor = 8, channel_min = None,
                 pad_type = "", act_layer = None,
                 norm_layer = nn.BatchNorm2d, norm_kwargs = None):
        self.channel_multiplier = channel_multiplier
        self.channel_divisor = channel_divisor
        self.channel_min = channel_min
        self.pad_type = pad_type
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.norm_kwargs = norm_kwargs

        assert(self.act_layer is not None)

        # updated during build
        self.in_channels = None

    def MakeBlock(self, block_def):
        if block_def.block_type == "ir":
            block = InvertedResidual(block_def, self.in_channels, self.pad_type, self.act_layer, self.norm_layer, self.norm_kwargs)
        elif block_def.block_type == "ds":
            block = DepthwiseSeparableConv(block_def, self.in_channels, self.pad_type, self.act_layer, self.norm_layer, self.norm_kwargs)
        else:
            print(f"Uknkown block type ({block_type}) while building model.")
            assert(False)
        return block

    def MakeStack(self, stack_args):
        blocks = []
        # each stack (stage) contains a list of block arguments
        for i, block_args in enumerate(stack_args):
            block_def = deepcopy(block_args)
            block_def.out_channels = RoundChannels(block_args.out_channels, self.channel_multiplier, self.channel_divisor, self.channel_min)
            if i > 0:
                # only the first block in any stack can have a stride > 1
                block_def.stride = 1
            blocks.append(self.MakeBlock(block_def))
            self.in_channels = block_def.out_channels  # update in_channels for arg of next block
        return nn.Sequential(*blocks)

    def __call__(self, in_channels, block_args):
        self.in_channels = in_channels
        blocks = []
        # outer list of block_args defines the stacks ("stages" by some conventions)
        for stack in block_args:
            assert(isinstance(stack, list))
            blocks.append(self.MakeStack(stack))
        return blocks

def ParseKSize(ss):
    if ss.isdigit():
        return int(ss)
    else:
        return [int(k) for k in ss.split(".")]

def ScaleStageDepth(block_args : BlockDefine, num_repeat : int, depth_multiplier : float = 1.0, depth_trunc : str = "ceil"):
    """ Per-stage depth scaling
    Scales the block repeats in each stage. This depth scaling impl maintains
    compatibility with the EfficientNet scaling method, while allowing sensible
    scaling for other models that may have multiple block arg definitions in each stage.
    """

    # We scale the total repeat count for each stage, there may be multiple
    # block arg defs per stage so we need to sum.
    if depth_trunc == "round":
        # Truncating to int by rounding allows stages with few repeats to remain
        # proportionally smaller for longer. This is a good choice when stage definitions
        # include single repeat stages that we'd prefer to keep that way as long as possible
        num_repeat_scaled = max(1, round(num_repeat * depth_multiplier))
    else:
        # The default for EfficientNet truncates repeats to int via "ceil".
        # Any multiplier > 1.0 will result in an increased depth for every stage.
        num_repeat_scaled = int(math.ceil(num_repeat * depth_multiplier))

    # Proportionally distribute repeat count scaling to each block definition in the stage.
    # Allocation is done in reverse as it results in the first block being less likely to be scaled.
    # The first block makes less sense to repeat in most of the arch definitions.
    repeats_scaled = max(1, num_repeat_scaled)

    # Apply the calculated scaling to each block arg in the stage
    sa_scaled = []
    for i in range(repeats_scaled):
        sa_scaled.append(block_args)
    return sa_scaled

def DecodeArchDef(arch_defs, depth_multiplier = 1.0, depth_trunc = "ceil", fix_first_last = False):
    arch_args = []
    for stack_idx, arch_def in enumerate(arch_defs):
        if fix_first_last and ((stack_idx == 0) or (stack_idx == len(arch_defs) - 1)):
            dm = 1.0
        else:
            dm = depth_multiplier
        arch_args.append(ScaleStageDepth(arch_def.block_def, arch_def.num_repeat, dm, depth_trunc))
    return arch_args

def InitializeWeightTf(model, name = "", fix_group_fanout = True):
    # weight init as per Tensorflow Official impl
    # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    if isinstance(model, nn.Conv2d):
        fan_out = model.kernel_size[0] * model.kernel_size[1] * model.out_channels
        if fix_group_fanout:
            fan_out //= model.groups
        model.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if model.bias is not None:
            model.bias.data.zero_()
    elif isinstance(model, nn.BatchNorm2d):
        model.weight.data.fill_(1.0)
        model.bias.data.zero_()
    elif isinstance(model, nn.Linear):
        fan_out = model.weight.size(0)  # fan-out
        fan_in = 0
        if "routing_fn" in name:
            fan_in = model.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        model.weight.data.uniform_(-init_range, init_range)
        model.bias.data.zero_()

class EfficientNet(nn.Module):
    """ Generic EfficientNets

    An implementation of mobile optimized networks that covers:
      * EfficientNet (B0-B8, L2, CondConv, EdgeTPU)
      * MixNet (Small, Medium, and Large, XL)
      * MNASNet A1, B1, and small
      * FBNet C
      * Single-Path NAS Pixel1
    """

    def __init__(self, block_args, num_classes = 1000, in_channels = 3, num_features = 1280, stem_size = 32, fix_stem = False,
                 channel_multiplier = 1.0, channel_divisor = 8, channel_min = None, pad_type = "", act_layer = nn.ReLU,
                 norm_layer = nn.BatchNorm2d, norm_kwargs = None):
        super(EfficientNet, self).__init__()

        if not fix_stem:
            stem_size = RoundChannels(stem_size, channel_multiplier, channel_divisor, channel_min)
        self.conv_stem = SelectConv2d(in_channels, stem_size, 3, stride = 2, padding = pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace = True)
        in_channels = stem_size

        builder = EfficientNetBuilder(
            channel_multiplier, channel_divisor, channel_min,
            pad_type, act_layer, norm_layer, norm_kwargs)
        self.blocks = nn.Sequential(*builder(in_channels, block_args))
        in_channels = builder.in_channels

        self.conv_head = SelectConv2d(in_channels, num_features, 1, padding = pad_type)
        self.bn2 = norm_layer(num_features, **norm_kwargs)
        self.act2 = act_layer(inplace = True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_features, num_classes)

        for name, model in self.named_modules():
            InitializeWeightTf(model, name)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

def TfEfficientNetLite3():
    """
    EfficientNet-Lite3. Tensorflow compatible variant
    Creates an EfficientNet-Lite model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
    Paper: https://arxiv.org/abs/1905.11946
    """

    arch_defs = (
        ArchDefine(BlockDefine("ds", 3, 1, 1.0, 16), 1),
        ArchDefine(BlockDefine("ir", 3, 2, 6.0, 24), 2),
        ArchDefine(BlockDefine("ir", 5, 2, 6.0, 40), 2),
        ArchDefine(BlockDefine("ir", 3, 2, 6.0, 80), 3),
        ArchDefine(BlockDefine("ir", 5, 1, 6.0, 112), 3),
        ArchDefine(BlockDefine("ir", 5, 2, 6.0, 192), 4),
        ArchDefine(BlockDefine("ir", 3, 1, 6.0, 320), 1),
    )
    return EfficientNet(
        block_args = DecodeArchDef(arch_defs, depth_multiplier = 1.4, fix_first_last = True),
        num_features = 1280,
        stem_size = 32,
        fix_stem = True,
        channel_multiplier = 1.2,
        act_layer = nn.ReLU6,
        norm_kwargs = {"eps" : BatchNormEpsTfDefault},
        pad_type = "same",
    )
