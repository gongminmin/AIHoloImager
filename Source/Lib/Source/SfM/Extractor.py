# Copyright (c) 2026 Minmin Gong
#

import math
from pathlib import Path

import torch

from PythonSystem import ComputeDevice, DeviceSync, PurgeTorchCache, TensorToBytes
from LightGlue import SuperPoint

class Extractor:
    def __init__(self) -> None:
        self.device = ComputeDevice()

        this_py_dir = Path(__file__).parent.resolve()

        self.extractor = SuperPoint.FromPretrained(this_py_dir / "Models/LightGlue/", max_num_keypoints = 4096)
        self.extractor.eval()
        self.extractor = self.extractor.to(self.device)

    def Destroy(self) -> None:
        del self.extractor
        PurgeTorchCache()

    @torch.no_grad()
    def Extract(self, gray_image: torch.Tensor) -> dict[str, torch.Tensor]:
        gray_image = gray_image.squeeze(0).permute(2, 0, 1)
        gray_image = gray_image.to(torch.float32) / 255

        features = self.extractor.Extract(gray_image)
        DeviceSync(self.device)
        return features

    def ExportFeatures(self, features: dict[str, torch.Tensor]) -> tuple[bytes, bytes, bytes]:
        return (TensorToBytes(features["descriptors"]), TensorToBytes(features["keypoints"]), TensorToBytes(features["image_size"]))
