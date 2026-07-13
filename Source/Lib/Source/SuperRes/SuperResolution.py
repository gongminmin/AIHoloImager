# Copyright (c) 2026 Minmin Gong
#

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from RealESrGan import SrVggNetCompact
from PythonSystem import ComputeDevice, DeviceSync, PurgeTorchCache

class SuperResolution:
    def __init__(self):
        this_py_dir = Path(__file__).parent.resolve()

        self.device = ComputeDevice()

        model_path = this_py_dir / "Models/Real-ESRGAN/realesr-general-x4v3.pth"
        wdn_model_path = this_py_dir / "Models/Real-ESRGAN/realesr-general-wdn-x4v3.pth"

        denoise_strength = 0.5

        model = SrVggNetCompact.FromPretrained(paths = (model_path, wdn_model_path), dni_weights = (denoise_strength, 1 - denoise_strength))
        self.model = model.to(self.device).to(torch.float16)

        self.pixel_shuffle = nn.PixelShuffle(4)

    def Destroy(self) -> None:
        del self.model
        PurgeTorchCache()

    @torch.no_grad()
    def Process(self, image: torch.Tensor, ignore_alpha: Optional[bool] = False) -> torch.Tensor:
        num_channels = image.shape[-1]
        image = image.permute(0, 3, 1, 2)
        image = image.to(torch.float16).to(self.device).contiguous()
        image /= 255.0

        alpha = None
        if num_channels == 4:
            rgb = image[:, 0 : 3, :, :]
            if not ignore_alpha:
                alpha = image[:, 3 : 4, :, :]
        elif num_channels == 3:
            rgb = image
        elif num_channels == 1:
            rgb = image.repeat(1, 3, 1, 1)

        output = self.model(rgb)

        output_alpha = None
        if alpha is None:
            if num_channels == 4:
                output_alpha = torch.ones((output.shape[0], 1, output.shape[2], output.shape[3], output.shape[4]), device = self.device)
        else:
            alpha = alpha.repeat(1, 3, 1, 1)
            output_alpha = self.model(alpha)
            output_alpha = output_alpha[:, 0 : 1, :, :, :]

        if output_alpha is not None:
            output = torch.cat((output, output_alpha), 1)

        output = output.squeeze(0)
        output = ((output * 0.5 + 0.5) * 255.0).clamp(0, 255).to(torch.uint8)
        output = output.permute(1, 2, 3, 0).contiguous()

        DeviceSync(self.device)
        return output
