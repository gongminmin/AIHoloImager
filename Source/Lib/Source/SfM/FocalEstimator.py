# Copyright (c) 2025 Minmin Gong
#

from pathlib import Path

import torch

from PythonSystem import ComputeDevice, PurgeTorchCache, TensorFromBytes
from MoGe import MoGeModel

class FocalEstimator:
    def __init__(self):
        self.device = ComputeDevice()

        this_py_dir = Path(__file__).parent.resolve()

        self.model = MoGeModel.FromPretrained(this_py_dir / "Models/moge-vitl/model.pt")
        self.model.eval()
        self.model = self.model.to(self.device)

    def Destroy(self):
        del self.model
        PurgeTorchCache()

    @torch.no_grad()
    def Process(self, image, image_width, image_height, channels):
        image = TensorFromBytes(image, torch.uint8, image_height * image_width * channels, self.device)
        image = image.reshape(image_height, image_width, channels)[:, :, 0 : 3].permute(2, 0, 1)

        image = image.to(torch.float16).contiguous()
        image /= 255.0

        return self.model.Focal(image)
