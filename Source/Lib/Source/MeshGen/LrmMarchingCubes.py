# Copyright (c) 2024 Minmin Gong
#

# Take InstantMesh/src/models/lrm.py as a reference

import mcubes
import nvdiffrast.torch as dr
import torch
import torch.nn as nn

from src.models.decoder.transformer import TriplaneTransformer
from src.models.encoder.dino_wrapper import DinoWrapper
from src.models.renderer.synthesizer import TriplaneSynthesizer
from src.utils.mesh_util import xatlas_uvmap

class LrmMarchingCubes(nn.Module):
    def __init__(self,
                 device,
                 encoder_freeze: bool = False,
                 encoder_model_name: str = 'facebook/dino-vitb16',
                 encoder_feat_dim: int = 768,
                 transformer_dim: int = 1024,
                 transformer_layers: int = 16,
                 transformer_heads: int = 16,
                 triplane_low_res: int = 32,
                 triplane_high_res: int = 64,
                 triplane_dim: int = 80,
                 rendering_samples_per_ray: int = 128,
                 grid_res: int = 256,
                 mesh_threshold: float = 10.0):
        super(LrmMarchingCubes, self).__init__()

        self.device = device
        self.grid_res = grid_res
        self.mesh_threshold = mesh_threshold

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

    def PreTrainedModelName(self):
        return "instant_nerf_large"

    def GenerateMesh(self, images, cameras, texture_resolution: int = 1024):
        image_feats = self.encoder(images.unsqueeze(0), cameras.unsqueeze(0))
        image_feats = image_feats.reshape(1, image_feats.shape[0] * image_feats.shape[1], image_feats.shape[2])

        planes = self.transformer(image_feats)
        assert(planes.shape[0] == 1)
        planes = planes.squeeze(0)

        grid_out = self.synthesizer.forward_grid(
            planes = planes.unsqueeze(0),
            grid_size = self.grid_res,
        )["sigma"]
        assert(grid_out.shape[0] == 1)
        grid_out = grid_out.squeeze(0)

        sdf = grid_out.squeeze(-1)
        vertices, faces = mcubes.marching_cubes(sdf.cpu().numpy(), self.mesh_threshold)

        vertices = torch.tensor(vertices, dtype = torch.float32, device = self.device)
        vertices = vertices / (self.grid_res - 1) * 2 - 1

        faces = torch.tensor(faces.astype(int), dtype = torch.long, device = self.device)

        ctx = dr.RasterizeCudaContext(device = self.device)
        uvs, mesh_tex_idx, gb_pos, tex_hard_mask = xatlas_uvmap(
            ctx, vertices, faces, resolution = texture_resolution)
        tex_hard_mask = tex_hard_mask.float().squeeze(0)
        texture_map = self.PredictTexture(planes, gb_pos, tex_hard_mask) * tex_hard_mask

        return vertices, faces, uvs, mesh_tex_idx, texture_map

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

        tex_feat = self.synthesizer.forward_points(
            planes.unsqueeze(0),
            tex_pos.unsqueeze(0)
        )["rgb"]
        assert(tex_feat.shape[0] == 1)
        tex_feat = tex_feat.squeeze(0)

        final_tex_feat = torch.zeros(hard_mask.shape[0] * hard_mask.shape[1], tex_feat.shape[-1], device = self.device)
        expanded_hard_mask = (hard_mask.reshape(-1, 1) > 0.5).expand(-1, final_tex_feat.shape[-1])
        final_tex_feat[expanded_hard_mask] = tex_feat[: n_point_list].reshape(-1)

        return final_tex_feat.reshape(hard_mask.shape[0], hard_mask.shape[1], final_tex_feat.shape[-1])
