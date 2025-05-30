# Copyright (c) 2025 Minmin Gong
#

# From https://github.com/xuebinqin/U-2-Net/blob/master/model/u2net.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as functional

class ReBnConv(nn.Module):
    def __init__(self, in_ch = 3, out_ch = 3, dirate = 1, device : Optional[torch.device] = None):
        super(ReBnConv, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding = 1 * dirate, dilation = 1 * dirate, device = device)
        self.bn_s1 = nn.BatchNorm2d(out_ch, device = device)
        self.relu_s1 = nn.ReLU(inplace = True)

    def forward(self,x):
        hx = x
        hx = self.conv_s1(hx)
        hx = self.bn_s1(hx)
        xout = self.relu_s1(hx)
        return xout

def UpsampleLike(src, target):
    return functional.interpolate(src, size = target.shape[2 : ], mode = "bilinear")

class Rsu4(nn.Module):
    def __init__(self, in_ch = 3, mid_ch = 12, out_ch = 3, device : Optional[torch.device] = None):
        super(Rsu4, self).__init__()

        self.rebnconvin = ReBnConv(in_ch, out_ch, dirate = 1, device = device)

        self.rebnconv1 = ReBnConv(out_ch, mid_ch, dirate = 1, device = device)
        self.pool1 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.rebnconv2 = ReBnConv(mid_ch, mid_ch, dirate = 1, device = device)
        self.pool2 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.rebnconv3 = ReBnConv(mid_ch, mid_ch, dirate = 1, device = device)

        self.rebnconv4 = ReBnConv(mid_ch, mid_ch, dirate = 2, device = device)

        self.rebnconv3d = ReBnConv(mid_ch * 2, mid_ch, dirate = 1, device = device)
        self.rebnconv2d = ReBnConv(mid_ch * 2, mid_ch, dirate = 1, device = device)
        self.rebnconv1d = ReBnConv(mid_ch * 2, out_ch, dirate = 1, device = device)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = UpsampleLike(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = UpsampleLike(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

class Rsu4F(nn.Module):
    def __init__(self, in_ch = 3, mid_ch = 12, out_ch = 3, device : Optional[torch.device] = None):
        super(Rsu4F, self).__init__()

        self.rebnconvin = ReBnConv(in_ch, out_ch, dirate = 1, device = device)

        self.rebnconv1 = ReBnConv(out_ch, mid_ch, dirate = 1, device = device)
        self.rebnconv2 = ReBnConv(mid_ch, mid_ch, dirate = 2, device = device)
        self.rebnconv3 = ReBnConv(mid_ch, mid_ch, dirate = 4, device = device)

        self.rebnconv4 = ReBnConv(mid_ch, mid_ch, dirate = 8, device = device)

        self.rebnconv3d = ReBnConv(mid_ch * 2, mid_ch, dirate = 4, device = device)
        self.rebnconv2d = ReBnConv(mid_ch * 2, mid_ch, dirate = 2, device = device)
        self.rebnconv1d = ReBnConv(mid_ch * 2, out_ch, dirate = 1, device = device)

    def forward(self,x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin

class Rsu5(nn.Module):
    def __init__(self, in_ch = 3, mid_ch = 12, out_ch = 3, device : Optional[torch.device] = None):
        super(Rsu5, self).__init__()

        self.rebnconvin = ReBnConv(in_ch, out_ch, dirate = 1, device = device)

        self.rebnconv1 = ReBnConv(out_ch, mid_ch, dirate = 1, device = device)
        self.pool1 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.rebnconv2 = ReBnConv(mid_ch, mid_ch, dirate = 1, device = device)
        self.pool2 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.rebnconv3 = ReBnConv(mid_ch, mid_ch, dirate = 1, device = device)
        self.pool3 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.rebnconv4 = ReBnConv(mid_ch, mid_ch, dirate = 1, device = device)

        self.rebnconv5 = ReBnConv(mid_ch, mid_ch, dirate = 2, device = device)

        self.rebnconv4d = ReBnConv(mid_ch * 2, mid_ch, dirate = 1, device = device)
        self.rebnconv3d = ReBnConv(mid_ch * 2, mid_ch, dirate = 1, device = device)
        self.rebnconv2d = ReBnConv(mid_ch * 2, mid_ch, dirate = 1, device = device)
        self.rebnconv1d = ReBnConv(mid_ch * 2, out_ch, dirate = 1, device = device)

    def forward(self,x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = UpsampleLike(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = UpsampleLike(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = UpsampleLike(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

class Rsu6(nn.Module):
    def __init__(self, in_ch = 3, mid_ch = 12, out_ch = 3, device : Optional[torch.device] = None):
        super(Rsu6, self).__init__()

        self.rebnconvin = ReBnConv(in_ch, out_ch, dirate = 1, device = device)

        self.rebnconv1 = ReBnConv(out_ch, mid_ch, dirate = 1, device = device)
        self.pool1 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.rebnconv2 = ReBnConv(mid_ch, mid_ch, dirate = 1, device = device)
        self.pool2 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.rebnconv3 = ReBnConv(mid_ch, mid_ch, dirate = 1, device = device)
        self.pool3 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.rebnconv4 = ReBnConv(mid_ch, mid_ch, dirate = 1, device = device)
        self.pool4 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.rebnconv5 = ReBnConv(mid_ch, mid_ch, dirate = 1, device = device)

        self.rebnconv6 = ReBnConv(mid_ch, mid_ch, dirate = 2, device = device)

        self.rebnconv5d = ReBnConv(mid_ch * 2, mid_ch, dirate = 1, device = device)
        self.rebnconv4d = ReBnConv(mid_ch * 2, mid_ch, dirate = 1, device = device)
        self.rebnconv3d = ReBnConv(mid_ch * 2, mid_ch, dirate = 1, device = device)
        self.rebnconv2d = ReBnConv(mid_ch * 2, mid_ch, dirate = 1, device = device)
        self.rebnconv1d = ReBnConv(mid_ch * 2, out_ch, dirate = 1, device = device)

    def forward(self,x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = UpsampleLike(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = UpsampleLike(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = UpsampleLike(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = UpsampleLike(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

class Rsu7(nn.Module):
    def __init__(self, in_ch = 3, mid_ch = 12, out_ch = 3, device : Optional[torch.device] = None):
        super(Rsu7, self).__init__()

        self.rebnconvin = ReBnConv(in_ch, out_ch, dirate = 1, device = device)

        self.rebnconv1 = ReBnConv(out_ch, mid_ch, dirate = 1, device = device)
        self.pool1 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.rebnconv2 = ReBnConv(mid_ch, mid_ch, dirate = 1, device = device)
        self.pool2 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.rebnconv3 = ReBnConv(mid_ch, mid_ch, dirate = 1, device = device)
        self.pool3 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.rebnconv4 = ReBnConv(mid_ch, mid_ch, dirate = 1, device = device)
        self.pool4 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.rebnconv5 = ReBnConv(mid_ch, mid_ch, dirate = 1, device = device)
        self.pool5 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.rebnconv6 = ReBnConv(mid_ch, mid_ch, dirate = 1, device = device)

        self.rebnconv7 = ReBnConv(mid_ch, mid_ch, dirate = 2, device = device)

        self.rebnconv6d = ReBnConv(mid_ch * 2, mid_ch, dirate = 1, device = device)
        self.rebnconv5d = ReBnConv(mid_ch * 2, mid_ch, dirate = 1, device = device)
        self.rebnconv4d = ReBnConv(mid_ch * 2, mid_ch, dirate = 1, device = device)
        self.rebnconv3d = ReBnConv(mid_ch * 2, mid_ch, dirate = 1, device = device)
        self.rebnconv2d = ReBnConv(mid_ch * 2, mid_ch, dirate = 1, device = device)
        self.rebnconv1d = ReBnConv(mid_ch * 2, out_ch, dirate = 1, device = device)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = UpsampleLike(hx6d, hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = UpsampleLike(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = UpsampleLike(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = UpsampleLike(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = UpsampleLike(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

class U2Net(nn.Module):
    def __init__(self, in_ch = 3,out_ch = 1, device : Optional[torch.device] = None):
        super(U2Net, self).__init__()

        self.stage1 = Rsu7(in_ch, 32, 64, device = device)
        self.pool12 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.stage2 = Rsu6(64, 32, 128, device = device)
        self.pool23 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.stage3 = Rsu5(128, 64, 256, device = device)
        self.pool34 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.stage4 = Rsu4(256, 128, 512, device = device)
        self.pool45 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.stage5 = Rsu4F(512, 256, 512, device = device)
        self.pool56 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.stage6 = Rsu4F(512, 256, 512, device = device)

        self.stage5d = Rsu4F(1024, 256, 512, device = device)
        self.stage4d = Rsu4(1024, 128, 256, device = device)
        self.stage3d = Rsu5(512, 64, 128, device = device)
        self.stage2d = Rsu6(256, 32, 64, device = device)
        self.stage1d = Rsu7(128, 16, 64, device = device)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding = 1, device = device)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding = 1, device = device)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding = 1, device = device)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding = 1, device = device)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding = 1, device = device)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding = 1, device = device)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1, device = device)

    def forward(self, x):
        hx = x

        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)
        hx6up = UpsampleLike(hx6, hx5)

        # Decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = UpsampleLike(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = UpsampleLike(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = UpsampleLike(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = UpsampleLike(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = UpsampleLike(d2, d1)

        d3 = self.side3(hx3d)
        d3 = UpsampleLike(d3, d1)

        d4 = self.side4(hx4d)
        d4 = UpsampleLike(d4, d1)

        d5 = self.side5(hx5d)
        d5 = UpsampleLike(d5, d1)

        d6 = self.side6(hx6)
        d6 = UpsampleLike(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return functional.sigmoid(d0)

class U2NetSmall(nn.Module):
    def __init__(self, in_ch = 3,out_ch = 1, device : Optional[torch.device] = None):
        super(U2NetSmall, self).__init__()

        self.stage1 = Rsu7(in_ch, 16, 64, device = device)
        self.pool12 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.stage2 = Rsu6(64, 16, 64, device = device)
        self.pool23 = nn.MaxPool2d(2, stride = 2,ceil_mode = True)

        self.stage3 = Rsu5(64, 16, 64, device = device)
        self.pool34 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.stage4 = Rsu4(64, 16, 64, device = device)
        self.pool45 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.stage5 = Rsu4F(64, 16, 64, device = device)
        self.pool56 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)

        self.stage6 = Rsu4F(64, 16, 64, device = device)

        self.stage5d = Rsu4F(128, 16, 64)
        self.stage4d = Rsu4(128, 16, 64)
        self.stage3d = Rsu5(128, 16, 64)
        self.stage2d = Rsu6(128, 16, 64)
        self.stage1d = Rsu7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding = 1, device = device)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding = 1, device = device)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding = 1, device = device)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding = 1, device = device)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding = 1, device = device)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding = 1, device = device)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1, device = device)

    def forward(self, x):
        hx = x

        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)
        hx6up = UpsampleLike(hx6, hx5)

        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = UpsampleLike(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = UpsampleLike(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = UpsampleLike(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = UpsampleLike(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = UpsampleLike(d2, d1)

        d3 = self.side3(hx3d)
        d3 = UpsampleLike(d3, d1)

        d4 = self.side4(hx4d)
        d4 = UpsampleLike(d4, d1)

        d5 = self.side5(hx5d)
        d5 = UpsampleLike(d5, d1)

        d6 = self.side6(hx6)
        d6 = UpsampleLike(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return functional.sigmoid(d0)
