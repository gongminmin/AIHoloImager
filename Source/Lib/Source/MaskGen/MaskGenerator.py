# Copyright (c) 2024-2025 Minmin Gong
#

from pathlib import Path

import numpy as np
import torch

from PythonSystem import ComputeDevice, GeneralDevice
from U2Net import U2Net

class MaskGenerator:
    def __init__(self):
        this_py_dir = Path(__file__).parent.resolve()

        self.device = ComputeDevice()

        u2net_model_path = this_py_dir / "Models/U-2-Net/u2net.pth"
        self.u2net = U2Net(3, 1)
        self.u2net.load_state_dict(torch.load(u2net_model_path, map_location = GeneralDevice(), weights_only = True))
        self.u2net.eval()
        self.u2net.to(self.device)

    def Destroy(self):
        del self.u2net
        torch.cuda.empty_cache()

    @torch.no_grad()
    def Gen(self, img_data : bytes, width : int, height : int, num_channels : int) -> bytes:
        norm_img = np.frombuffer(img_data, dtype = np.float32, count = width * height * num_channels)
        norm_img = torch.from_numpy(norm_img.copy()).to(self.device)
        norm_img = norm_img.reshape(1, num_channels, height, width)

        pred = self.u2net(norm_img).squeeze(0)

        pred = pred.cpu().numpy()
        return pred.tobytes()
