# Copyright (c) 2024-2025 Minmin Gong
#

from pathlib import Path

import torch

from PythonSystem import ComputeDevice, PurgeTorchCache
from Trellis.Pipelines import TrellisImageTo3DPipeline

class MeshGenerator:
    def __init__(self, gpu_system):
        this_py_dir = Path(__file__).parent.resolve()

        self.device = ComputeDevice()
        self.gpu_system = gpu_system

        pretrained_dir = this_py_dir / "Models/TRELLIS-image-large"
        self.pipeline = TrellisImageTo3DPipeline.FromPretrained(pretrained_dir)
        self.pipeline.to(self.device)

    def Destroy(self):
        del self.pipeline
        PurgeTorchCache()

    @torch.no_grad()
    def GenFeatures(self, images):
        torch_images = []
        for image in images:
            image = image.squeeze(0).permute(2, 0, 1)
            image = image[0 : 3, :, :].float() / 255
            torch_images.append(image)
        images = torch.stack(torch_images).contiguous()

        steps = max(images.shape[0] * 3, 25)
        sparse_volume = self.pipeline.Run(
            self.gpu_system,
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
        self.coords = sparse_volume.coords

        DensitySize = 8 * 1
        DeformationSize = 8 * 3
        WeightSize = 21
        AttributeSize = 8 * (3 + 3)

        start_offset = 0;

        self.density_features = sparse_volume.feats[:, start_offset : start_offset + DensitySize]
        density_bias = -1.0 / self.resolution
        self.density_features += density_bias
        start_offset += DensitySize

        self.deformation_features = sparse_volume.feats[:, start_offset : start_offset + DeformationSize]
        start_offset += DeformationSize

        start_offset += WeightSize

        indices = []
        for i in range(8):
            indices.append(start_offset + i * (3 + 3) + 0)
            indices.append(start_offset + i * (3 + 3) + 1)
            indices.append(start_offset + i * (3 + 3) + 2)
        indices = torch.tensor(indices, device = self.device)
        self.color_features = torch.index_select(sparse_volume.feats, -1, indices)

    def Resolution(self):
        return self.resolution

    def Coords(self) -> torch.Tensor:
        return self.coords

    def DensityFeatures(self) -> torch.Tensor:
        return self.density_features

    def DeformationFeatures(self) -> torch.Tensor:
        return self.deformation_features

    def ColorFeatures(self) -> torch.Tensor:
        return self.color_features
