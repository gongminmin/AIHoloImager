# Copyright (c) 2024-2025 Minmin Gong
#

from pathlib import Path

import torch

from PythonSystem import ComputeDevice, PurgeTorchCache, TensorFromBytes, TensorToBytes
from Trellis.Pipelines import TrellisImageTo3DPipeline

class MeshGenerator:
    def __init__(self):
        this_py_dir = Path(__file__).parent.resolve()

        self.device = ComputeDevice()

        pretrained_dir = this_py_dir / "Models/TRELLIS-image-large"
        self.pipeline = TrellisImageTo3DPipeline.FromPretrained(pretrained_dir)
        self.pipeline.to(self.device)

    def Destroy(self):
        del self.pipeline
        PurgeTorchCache()

    @torch.no_grad()
    def GenFeatures(self, images, width, height, num_channels):
        torch_images = []
        for image in images:
            image = TensorFromBytes(image, torch.uint8, height * width * num_channels, self.device)
            image = image.reshape(height, width, num_channels)
            image = image.permute(2, 0, 1)
            image = image[0 : 3, :, :].float() / 255
            torch_images.append(image)
        images = torch.stack(torch_images).contiguous()

        steps = max(images.shape[0] * 3, 25)
        sparse_volume = self.pipeline.Run(
            images,
            sparse_structure_sampler_params = {
                "steps" : steps,
                "cfg_strength" : 7.5,
            },
            slat_sampler_params = {
                "steps" : steps,
                "cfg_strength" : 3,
            },
        )[0]

        self.resolution = sparse_volume.resolution
        self.coords = TensorToBytes(sparse_volume.coords)

        DensitySize = 8 * 1
        DeformationSize = 8 * 3
        WeightSize = 21
        AttributeSize = 8 * (3 + 3)

        start_offset = 0;

        density_features = sparse_volume.feats[:, start_offset : start_offset + DensitySize]
        density_bias = -1.0 / self.resolution
        density_features += density_bias
        self.density_features = TensorToBytes(density_features)
        start_offset += DensitySize

        deformation_features = sparse_volume.feats[:, start_offset : start_offset + DeformationSize]
        self.deformation_features = TensorToBytes(deformation_features)
        start_offset += DeformationSize

        start_offset += WeightSize

        indices = []
        for i in range(8):
            indices.append(start_offset + i * (3 + 3) + 0)
            indices.append(start_offset + i * (3 + 3) + 1)
            indices.append(start_offset + i * (3 + 3) + 2)
        indices = torch.tensor(indices, device = self.device)
        color_features = torch.index_select(sparse_volume.feats, -1, indices)
        self.color_features = TensorToBytes(color_features)

    def Resolution(self):
        return self.resolution

    def Coords(self):
        return self.coords

    def DensityFeatures(self):
        return self.density_features

    def DeformationFeatures(self):
        return self.deformation_features

    def ColorFeatures(self):
        return self.color_features
