# Copyright (c) 2026 Minmin Gong
#

import math
from pathlib import Path

import torch

from LightGlue import LightGlue
from PythonSystem import ComputeDevice, PurgeTorchCache, TensorFromBytes, TensorToBytes

class Matcher:
    def __init__(self):
        self.device = ComputeDevice()

        this_py_dir = Path(__file__).parent.resolve()

        self.model = LightGlue.FromPretrained(this_py_dir / "Models/LightGlue/", features = "superpoint")
        self.model.eval()
        self.model = self.model.to(self.device)

    def Destroy(self):
        del self.model
        PurgeTorchCache()

    @torch.no_grad()
    def Match(self, features: tuple, image_idx0: int, image_idx1: int) -> tuple:
        num_images = len(features)

        features0 = features[image_idx0]
        features1 = features[image_idx1]

        matches = self.model({"image0": features0, "image1": features1})
        pairs = matches["matches"][0][..., 0 : 2].to(torch.int32)
        putative_matches = (TensorToBytes(pairs), pairs.shape[-2])

        PurgeTorchCache()

        return putative_matches
