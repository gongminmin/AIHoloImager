# Copyright (c) 2025-2026 Minmin Gong
#

from pathlib import Path

import torch

from PythonSystem import ComputeDevice, PurgeTorchCache
from MoGe import MoGeModel

class PointCloudEstimator:
    def __init__(self) -> None:
        self.device = ComputeDevice()

        this_py_dir = Path(__file__).parent.resolve()

        self.model = MoGeModel.FromPretrained(this_py_dir / "Models/moge-2-vitl/model.pt")
        self.model.eval()
        self.model = self.model.to(self.device)
        self.model = self.model.to(torch.float16)

    def Destroy(self):
        del self.model
        PurgeTorchCache()

    @torch.no_grad()
    def Focal(self, image: torch.Tensor) -> float:
        image = image.squeeze(0)[..., 0 : 3].permute(2, 0, 1)

        image = image.to(torch.float16).contiguous()
        image /= 255.0

        return self.model.Focal(image)

    @torch.no_grad()
    def PointCloud(self, image: torch.Tensor, fov_x: float) -> torch.Tensor:
        image = image.squeeze(0)[..., 0 : 3].permute(2, 0, 1)

        image = image.to(torch.float16).contiguous()
        image /= 255.0

        return self.model.PointCloud(image, fov_x)
