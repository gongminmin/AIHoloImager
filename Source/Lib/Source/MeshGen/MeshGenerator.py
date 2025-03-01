# Copyright (c) 2024-2025 Minmin Gong
#

from pathlib import Path

import numpy as np
from PIL import Image
import torch

import Util

from Trellis.Pipelines import TrellisImageTo3DPipeline

class MeshGenerator:
    def __init__(self):
        this_py_dir = Path(__file__).parent.resolve()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pretrained_dir = this_py_dir.joinpath("Models/TRELLIS-image-large")
        reload_from_network = True
        if pretrained_dir.exists():
            print(f"Load from {pretrained_dir}.")
            try:
                self.pipeline = TrellisImageTo3DPipeline.FromPretrained(pretrained_dir, None)
                reload_from_network = False
            except Exception as e:
                print(f"Failed. Retry from network. ", e)

        if reload_from_network:
            self.pipeline = TrellisImageTo3DPipeline.FromPretrained("JeffreyXiang/TRELLIS-image-large", pretrained_dir)

        self.pipeline.to(self.device)

    def Destroy(self):
        del self.pipeline
        del self.device
        torch.cuda.empty_cache()

    @torch.no_grad()
    def GenVolume(self, images):
        pil_images = []
        for image in images:
            image_data = image[0]
            width = image[1]
            height = image[2]
            num_channels = image[3]
            if num_channels == 1:
                mode = "L"
            elif num_channels == 3:
                mode = "RGB"
            else:
                assert(num_channels == 4)
                mode = "RGBA"
            pil_images.append(Image.frombuffer(mode, (width, height), image_data))
    
        steps = max(len(pil_images) * 3, 25)
        sparse_volume = self.pipeline.Run(
            pil_images,
            seed = 1,
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
        self.coords = sparse_volume.coords.cpu().numpy()

        DensitySize = 8 * 1
        DeformationSize = 8 * 3
        WeightSize = 21
        AttributeSize = 8 * (3 + 3)

        start_offset = 0;

        self.density_features = sparse_volume.feats[:, start_offset : start_offset + DensitySize]
        density_bias = -1.0 / self.resolution
        self.density_features += density_bias
        self.density_features = self.density_features.cpu().numpy()
        start_offset += DensitySize

        self.deformation_features = sparse_volume.feats[:, start_offset : start_offset + DeformationSize]
        self.deformation_features = self.deformation_features.cpu().numpy()
        start_offset += DeformationSize

        start_offset += WeightSize

        indices = []
        for i in range(8):
            indices.append(start_offset + i * (3 + 3) + 0)
            indices.append(start_offset + i * (3 + 3) + 1)
            indices.append(start_offset + i * (3 + 3) + 2)
        indices = torch.tensor(indices, device = self.device)
        self.color_features = torch.index_select(sparse_volume.feats, -1, indices)
        self.color_features = self.color_features.cpu().numpy()

    def Resolution(self):
        return self.resolution

    def Coords(self):
        return self.coords.tobytes()

    def DensityFeatures(self):
        return self.density_features.tobytes()
        
    def DeformationFeatures(self):
        return self.deformation_features.tobytes()

    def ColorFeatures(self):
        return self.color_features.tobytes()
