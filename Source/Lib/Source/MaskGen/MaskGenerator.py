# Copyright (c) 2024-2025 Minmin Gong
#

from pathlib import Path

import torch
from torch.nn.utils import skip_init

from PythonSystem import ComputeDevice, GeneralDevice, PurgeTorchCache
from U2Net import U2Net, U2NetSmall

class MaskGenerator:
    def __init__(self):
        this_py_dir = Path(__file__).parent.resolve()

        self.device = ComputeDevice()

        u2net_model_path = this_py_dir / "Models/U-2-Net/u2net.pth"
        self.u2net = skip_init(U2Net, 3, 1)
        self.u2net.load_state_dict(torch.load(u2net_model_path, map_location = GeneralDevice(), weights_only = True))
        self.u2net.eval()
        self.u2net.to(self.device)

        u2net_small_model_path = this_py_dir / "Models/U-2-Net/u2netp.pth"
        self.u2net_small = skip_init(U2NetSmall, 3, 1)
        self.u2net_small.load_state_dict(torch.load(u2net_small_model_path, map_location = GeneralDevice(), weights_only = True))
        self.u2net_small.eval()
        self.u2net_small.to(self.device)

    def Destroy(self):
        del self.u2net
        del self.u2net_small
        PurgeTorchCache()

    @torch.no_grad()
    def Gen(self, norm_img: torch.Tensor, width: int, height: int, num_channels: int, large_model: bool) -> torch.Tensor:
        norm_img = norm_img.reshape(1, num_channels, height, width)

        if large_model:
            pred = self.u2net(norm_img)
        else:
            pred = self.u2net_small(norm_img)

        pred = pred.reshape(height, width, 1)
        return pred
