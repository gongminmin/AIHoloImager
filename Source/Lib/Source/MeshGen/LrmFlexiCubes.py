# Copyright (c) 2024 Minmin Gong
#

# Take InstantMesh/src/models/lrm_mesh.py as a reference
# The major difference is removing the batch_size, since it's always 1.
# Also, tons of code clean and simplification is taken placed.

import torch
import torch.nn as nn

from src.models.decoder.transformer import TriplaneTransformer
from src.models.encoder.dino_wrapper import DinoWrapper
from src.models.geometry.camera.perspective_camera import PerspectiveCamera
from src.models.geometry.render.neural_render import NeuralRender
from src.models.geometry.rep_3d.flexicubes_geometry import FlexiCubesGeometry
from src.models.renderer.synthesizer_mesh import TriplaneSynthesizer
from src.utils.mesh_util import xatlas_uvmap

class LrmFlexiCubes(nn.Module):
    def __init__(self,
                 device,
                 encoder_freeze: bool = False,
                 encoder_model_name: str = "facebook/dino-vitb16",
                 encoder_feat_dim: int = 768,
                 transformer_dim: int = 1024,
                 transformer_layers: int = 16,
                 transformer_heads: int = 16,
                 triplane_low_res: int = 32,
                 triplane_high_res: int = 64,
                 triplane_dim: int = 80,
                 rendering_samples_per_ray: int = 128,
                 grid_res: int = 128,
                 grid_scale: float = 2.0,
                 fovy = 50.0):
        super(LrmFlexiCubes, self).__init__()

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

        camera = PerspectiveCamera(fovy = fovy, device = self.device)
        renderer = NeuralRender(self.device, camera_model = camera)
        self.geometry = FlexiCubesGeometry(
            grid_res = grid_res,
            scale = grid_scale,
            renderer = renderer,
            render_type = "neural_render",
            device = self.device,
        )

    def PreTrainedModelName(self):
        return "instant_mesh_large"

    def GenerateMesh(self, images, cameras, texture_resolution: int = 1024):
        image_feats = self.encoder(images.unsqueeze(0), cameras.unsqueeze(0))
        image_feats = image_feats.reshape(1, image_feats.shape[0] * image_feats.shape[1], image_feats.shape[2])

        planes = self.transformer(image_feats)
        assert(planes.shape[0] == 1)
        planes = planes.squeeze(0)

        vertices, faces = self.PredictGeometry(planes)

        uvs, mesh_tex_idx, gb_pos, tex_hard_mask = xatlas_uvmap(
            self.geometry.renderer.ctx, vertices, faces, resolution = texture_resolution)
        tex_hard_mask = tex_hard_mask.float().squeeze(0)
        texture_map = self.PredictTexture(planes, gb_pos, tex_hard_mask) * tex_hard_mask

        return vertices, faces, uvs, mesh_tex_idx, texture_map

    def PredictSdfDeformation(self, planes):
        # Step 1: predict the SDF and deformation
        sdf, deformation, weight = self.synthesizer.get_geometry_prediction(
            planes.unsqueeze(0),
            self.geometry.verts.unsqueeze(0),
            self.geometry.indices
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
            new_sdf[:, self.geometry.center_indices] = (1.0 - sdf.min())  # greater than zero
            new_sdf[:, self.geometry.boundary_indices] = (-1 - sdf.max())  # smaller than zero
            update_mask = (new_sdf == 0).float()
            # Regulraization here is used to push the sdf to be a different sign (make it not fully positive or fully negative)
            sdf = torch.lerp(new_sdf, sdf, update_mask)

        return sdf, deformation, weight

    def PredictGeometry(self, planes):
        # Step 1: first get the sdf and deformation value for each vertices in the tetrahedon grid.
        sdf, deformation, weight = self.PredictSdfDeformation(planes)
        verts_deformed = self.geometry.verts + deformation
        indices = self.geometry.indices
        
        # Step 2: Using marching tet to obtain the mesh
        verts, faces, _ = self.geometry.get_mesh(
            verts_deformed,
            sdf.squeeze(-1),
            with_uv = False,
            indices = indices,
            weight_n = weight.squeeze(-1),
            is_training = False
        )
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
