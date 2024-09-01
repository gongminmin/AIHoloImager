# Copyright (c) 2024 Minmin Gong
#

# Take InstantMesh/src/models/lrm_mesh.py as a reference
# The major difference is
#   1. Removing the batch_size, since it's always 1.
#   2. Tons of code clean and simplification is taken placed.
#   3. Replace FlexiCubes with marching cubes.

from pathlib import Path

import mcubes
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
                                           dtype = torch.int)
        cube_corners_offset *= torch.tensor([size * size, size, 1], dtype = torch.int)
        cube_corners_offset = torch.sum(cube_corners_offset, 1)
        cube_corners_offset = cube_corners_offset.to(self.device)

        self.cube_indices = torch.arange(size * size * size, dtype = torch.int, device = self.device)
        self.cube_indices = self.cube_indices.reshape(size, size, size)
        self.cube_indices = self.cube_indices[0 : grid_res, 0 : grid_res, 0 : grid_res]
        self.cube_indices = self.cube_indices.reshape(-1, 1).expand(-1, 8)
        self.cube_indices = torch.add(self.cube_indices, cube_corners_offset)

        center = self.grid_res // 2 + 1
        self.cube_center_indices = torch.tensor([(center * size + center) * size + center],
                                                dtype = torch.int, device = self.device).unsqueeze(0)

        l = []
        for z in range(0, 2):
            for y in range(0, size):
                for x in range(0, size):
                    l.append((z * size + y) * size + x)
        for z in range(2, size - 2):
            for y in range(0, 2):
                for x in range(0, size):
                    l.append((z * size + y) * size + x)
            for y in range(2, size - 2):
                for x in range(0, 2):
                    l.append((z * size + y) * size + x)
                for x in range(size - 2, size):
                    l.append((z * size + y) * size + x)
            for y in range(size - 2, size):
                for x in range(0, size):
                    l.append((z * size + y) * size + x)
        for z in range(size - 2, size):
            for y in range(0, size):
                for x in range(0, size):
                    l.append((z * size + y) * size + x)
        self.cube_boundary_indices = torch.tensor(l, dtype = torch.int, device = self.device).unsqueeze(1)

    def GenerateMesh(self, images, cameras):
        image_feats = self.encoder(images.unsqueeze(0), cameras.unsqueeze(0))
        image_feats = image_feats.reshape(1, image_feats.shape[0] * image_feats.shape[1], image_feats.shape[2])

        planes = self.transformer(image_feats)
        assert(planes.shape[0] == 1)
        self.planes = planes.squeeze(0)

        return self.PredictGeometry(self.planes)

    def QueryColors(self, positions):
        colors = self.synthesizer.get_texture_prediction(self.planes.unsqueeze(0), positions.unsqueeze(0))
        assert(colors.shape[0] == 1)
        return colors.squeeze(0)

    def PredictSdfDeformation(self, planes):
        # Step 1: predict the SDF and deformation
        sdf, deformation, weight = self.synthesizer.get_geometry_prediction(
            planes.unsqueeze(0),
            self.cube_verts.unsqueeze(0),
            self.cube_indices
        )
        assert((sdf.shape[0] == 1) and (deformation.shape[0] == 1) and (weight.shape[0] == 1))
        sdf = sdf.squeeze(0)
        deformation = deformation.squeeze(0)
        weight = weight.squeeze(0)

        # Step 2: Normalize the deformation to avoid the flipped triangles.
        deformation_multiplier = 4.0
        deformation = 1.0 / (self.grid_res * deformation_multiplier) * torch.tanh(deformation)

        # Step 3: Fix some sdf if we observe empty shape (full positive or full negative)
        sdf_bxnxnxn = sdf.reshape(self.grid_res + 1, self.grid_res + 1, self.grid_res + 1)
        sdf_less_boundary = sdf_bxnxnxn[1 : -1, 1 : -1, 1 : -1].reshape(-1)
        pos_shape = torch.sum((sdf_less_boundary > 0).int(), dim = -1)
        neg_shape = torch.sum((sdf_less_boundary < 0).int(), dim = -1)
        zero_surface = torch.bitwise_or(pos_shape == 0, neg_shape == 0).item()
        if zero_surface:
            new_sdf = torch.zeros_like(sdf)
            new_sdf[:, self.cube_center_indices] = (1.0 - sdf.min())  # greater than zero
            new_sdf[:, self.cube_boundary_indices] = (-1 - sdf.max())  # smaller than zero
            update_mask = (new_sdf == 0).float()
            # Regulraization here is used to push the sdf to be a different sign (make it not fully positive or fully negative)
            sdf = torch.lerp(new_sdf, sdf, update_mask)

        return sdf, deformation, weight

    def PredictGeometry(self, planes):
        # Step 1: first get the sdf and deformation value for each vertices in the grid.
        sdf, deformation, weight = self.PredictSdfDeformation(planes)

        # Step 2: Using marching cubes to obtain the mesh
        sdf = sdf.squeeze(-1).reshape(self.grid_res + 1, self.grid_res + 1, self.grid_res + 1)
        verts, faces = mcubes.marching_cubes(sdf.cpu().numpy(), 0)

        verts = torch.tensor(verts, dtype = torch.float32, device = self.device)
        verts = (verts / self.grid_res - 0.5) * self.grid_scale

        faces = torch.tensor(faces.astype(int), dtype = torch.int32, device = self.device)
        # Flip the triangles
        faces = torch.index_select(faces, 1, torch.tensor([0, 2, 1], dtype = torch.int32, device = self.device))

        return verts, faces
