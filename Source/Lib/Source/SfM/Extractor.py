# Copyright (c) 2026 Minmin Gong
#

import math
from pathlib import Path

import torch

from PythonSystem import ComputeDevice, PurgeTorchCache, TensorFromBytes, TensorToBytes
from LightGlue import SuperPoint

class Extractor:
    def __init__(self):
        self.device = ComputeDevice()

        this_py_dir = Path(__file__).parent.resolve()

        self.extractor = SuperPoint.FromPretrained(this_py_dir / "Models/LightGlue/", max_num_keypoints = 4096)
        self.extractor.eval()
        self.extractor = self.extractor.to(self.device)

    def Destroy(self):
        del self.extractor
        PurgeTorchCache()

    @torch.no_grad()
    def Extract(self, image: bytes, image_width: int, image_height: int, channels: int) -> dict:
        image = TensorFromBytes(image, torch.uint8, image_height * image_width * channels, self.device)
        image = image.reshape(image_height, image_width, channels)[..., 0 : 3].permute(2, 0, 1)

        image = image.to(torch.float32) / 255

        scale = torch.tensor((0.299, 0.587, 0.114), device = self.device, dtype = image.dtype).view(1, 3, 1, 1)
        gray_image = (image * scale).sum(1, keepdim = True)

        return self.extractor.Extract(gray_image)

    def ExportFeatures(self, features: dict) -> tuple:
        return (TensorToBytes(features["descriptors"]), TensorToBytes(features["keypoints"]), TensorToBytes(features["image_size"]))
