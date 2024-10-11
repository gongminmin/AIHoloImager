# Copyright (c) 2024 Minmin Gong
#

# Take InstantMesh/src/models/lrm_mesh.py as a reference
# The major difference is
#   1. Removing the batch_size, since it's always 1.
#   2. Tons of code clean and simplification is taken placed.
#   3. Replace FlexiCubes with marching cubes.

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.decoder.transformer import TriplaneTransformer
from src.models.encoder.dino_wrapper import DinoWrapper
from src.models.renderer.synthesizer_mesh import TriplaneSynthesizer

class Lrm(nn.Module):
    def __init__(self,
                 device,
                 encoder_feat_dim : int = 768,
                 transformer_dim : int = 1024,
                 transformer_layers : int = 16,
                 transformer_heads : int = 16,
                 triplane_low_res : int = 32,
                 triplane_high_res : int = 64,
                 triplane_dim : int = 80,
                 rendering_samples_per_ray : int = 128,
                 grid_res : int = 128,
                 grid_scale : float = 2.0):
        super(Lrm, self).__init__()

        self.device = device
        self.grid_res = grid_res
        self.grid_scale = grid_scale

        this_py_dir = Path(__file__).parent.resolve()
        pretrained_dir = this_py_dir.joinpath("Models/dino-vitb16")
        reload_from_network = True
        if pretrained_dir.exists():
            print(f"Load from {pretrained_dir}.")
            try:
                self.encoder = DinoWrapper(
                    model_name = pretrained_dir,
                    freeze = False,
                )
                reload_from_network = False
            except:
                print(f"Failed. Retry from network.")

        if reload_from_network:
            self.encoder = DinoWrapper(
                model_name = "facebook/dino-vitb16",
                freeze = False,
            )

            pretrained_dir.mkdir(parents = True, exist_ok = True)
            self.encoder.model.save_pretrained(pretrained_dir)
            self.encoder.processor.save_pretrained(pretrained_dir)

        self.transformer = TriplaneTransformer(
            inner_dim = transformer_dim,
            num_layers = transformer_layers,
            num_heads = transformer_heads,
            image_feat_dim = encoder_feat_dim,
            triplane_low_res = triplane_low_res,
            triplane_high_res = triplane_high_res,
            triplane_dim = triplane_dim,
        )
 
        self.synthesizer = TriplaneSynthesizer(
            triplane_dim = triplane_dim,
            samples_per_ray = rendering_samples_per_ray,
        )

        size = self.grid_res + 1

        self.cube_verts = torch.nonzero(torch.ones((size, size, size), device = self.device)).float()
        self.cube_verts = (self.cube_verts / grid_res - 0.5) * grid_scale

        cube_corners_offset = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                                           dtype = torch.int32)
        cube_corners_offset *= torch.tensor([size * size, size, 1], dtype = torch.int32)
        cube_corners_offset = torch.sum(cube_corners_offset, 1)
        cube_corners_offset = cube_corners_offset.to(self.device)

        self.cube_indices = torch.arange(size * size * size, dtype = torch.int32, device = self.device)
        self.cube_indices = self.cube_indices.reshape(size, size, size)
        self.cube_indices = self.cube_indices[0 : grid_res, 0 : grid_res, 0 : grid_res]
        self.cube_indices = self.cube_indices.reshape(-1, 1).expand(-1, 8)
        self.cube_indices = torch.add(self.cube_indices, cube_corners_offset)

    def GenNeRF(self, images, cameras):
        image_feats = self.encoder(images.unsqueeze(0), cameras.unsqueeze(0))
        image_feats = image_feats.reshape(1, image_feats.shape[0] * image_feats.shape[1], image_feats.shape[2])

        planes = self.transformer(image_feats)
        assert(planes.shape[0] == 1)
        self.planes = planes.squeeze(0)

    def QuerySdfDeformation(self):
        sdf, deformation, weight = self.synthesizer.get_geometry_prediction(
            self.planes.unsqueeze(0),
            self.cube_verts.unsqueeze(0),
            self.cube_indices
        )
        assert((sdf.shape[0] == 1) and (deformation.shape[0] == 1))
        sdf = sdf.squeeze(0)
        deformation = deformation.squeeze(0)

        # Normalize the deformation to avoid the flipped triangles.
        deformation_multiplier = 4.0
        deformation = 1.0 / (self.grid_res * deformation_multiplier) * torch.tanh(deformation)

        size = self.grid_res + 1
        sdf = sdf.reshape(size, size, size, 1)
        deformation = deformation.reshape(size, size, size, -1)
        deformation *= size

        return torch.cat([sdf, deformation], dim = 3)

    def QueryColors(self, positions):
        colors = self.synthesizer.get_texture_prediction(self.planes.unsqueeze(0), positions.unsqueeze(0))
        assert(colors.shape[0] == 1)
        return colors.squeeze(0)
