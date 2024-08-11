# Copyright (c) 2024 Minmin Gong
#

# Take InstantMesh/src/models/lrm_mesh.py as a reference
# The major difference is
#   1. Removing the batch_size, since it's always 1.
#   2. Tons of code clean and simplification is taken placed.
#   3. Replace FlexiCubes with marching cubes.

import mcubes
import torch
import torch.nn as nn
import nvdiffrast.torch as dr

from src.models.decoder.transformer import TriplaneTransformer
from src.models.encoder.dino_wrapper import DinoWrapper
from src.models.renderer.synthesizer_mesh import TriplaneSynthesizer
from src.utils.mesh_util import xatlas_uvmap

class Lrm(nn.Module):
    def __init__(self,
                 device,
                 encoder_freeze : bool = False,
                 encoder_model_name : str = "facebook/dino-vitb16",
                 encoder_feat_dim : int = 768,
                 transformer_dim : int = 1024,
                 transformer_layers : int = 16,
                 transformer_heads : int = 16,
                 triplane_low_res : int = 32,
                 triplane_high_res : int = 64,
                 triplane_dim : int = 80,
                 rendering_samples_per_ray : int = 128,
                 grid_res : int = 128,
                 grid_scale : float = 2.0,
                 fovy : float = 50.0):
        super(Lrm, self).__init__()

        self.device = device
        self.grid_res = grid_res

        self.encoder = DinoWrapper(
            model_name = encoder_model_name,
            freeze = encoder_freeze,
        )

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

        self.renderer_ctx = dr.RasterizeCudaContext(device = self.device)

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

    def GenerateMesh(self, images, cameras, texture_resolution : int = 1024):
        image_feats = self.encoder(images.unsqueeze(0), cameras.unsqueeze(0))
        image_feats = image_feats.reshape(1, image_feats.shape[0] * image_feats.shape[1], image_feats.shape[2])

        planes = self.transformer(image_feats)
        assert(planes.shape[0] == 1)
        planes = planes.squeeze(0)

        vertices, faces = self.PredictGeometry(planes)

        uvs, mesh_tex_idx, gb_pos, tex_hard_mask = xatlas_uvmap(
            self.renderer_ctx, vertices, faces, resolution = texture_resolution)
        tex_hard_mask = tex_hard_mask.float().squeeze(0)
        texture_map = self.PredictTexture(planes, gb_pos, tex_hard_mask) * tex_hard_mask

        return vertices, faces, uvs, mesh_tex_idx, texture_map

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
        verts = verts / self.grid_res * 2 - 1
        verts *= 1.04 # Not sure why we need this scale

        faces = torch.tensor(faces.astype(int), dtype = torch.long, device = self.device)
        # Flip the triangles
        faces = torch.index_select(faces, 1, torch.tensor([0, 2, 1], dtype = torch.long, device = self.device))

        return verts, faces

    def PredictTexture(self, planes, tex_pos, hard_mask):
        tex_pos *= hard_mask
        tex_pos = tex_pos.reshape(-1, 3)

        n_point_list = torch.sum(hard_mask.long().reshape(-1), dim = -1)
        max_point = n_point_list.max()
        expanded_hard_mask = (hard_mask.reshape(-1, 1) > 0.5).expand(-1, 3)
        tex_pos = tex_pos[expanded_hard_mask].reshape(-1, 3)
        if tex_pos.shape[0] < max_point:
            tex_pos = torch.cat([tex_pos,
                    torch.zeros(max_point - tex_pos.shape[0], 3,
                            device = self.device, dtype = torch.float32)],
                    dim = 1)

        tex_feat = self.synthesizer.get_texture_prediction(planes.unsqueeze(0), tex_pos.unsqueeze(0))
        assert(tex_feat.shape[0] == 1)
        tex_feat = tex_feat.squeeze(0)

        final_tex_feat = torch.zeros(hard_mask.shape[0] * hard_mask.shape[1], tex_feat.shape[-1], device = self.device)
        expanded_hard_mask = (hard_mask.reshape(-1, 1) > 0.5).expand(-1, final_tex_feat.shape[-1])
        final_tex_feat[expanded_hard_mask] = tex_feat[: n_point_list].reshape(-1)

        return final_tex_feat.reshape(hard_mask.shape[0], hard_mask.shape[1], final_tex_feat.shape[-1])
