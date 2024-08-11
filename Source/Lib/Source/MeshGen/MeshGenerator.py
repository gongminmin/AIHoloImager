# Copyright (c) 2024 Minmin Gong
#

from pathlib import Path
import shutil

import importlib
from nvdiffrast.torch.ops import _cached_plugin

import numpy as np
import torch
from pytorch_lightning import seed_everything
from huggingface_hub import hf_hub_download

from src.utils.camera_util import get_zero123plus_input_cameras
from src.utils.mesh_util import save_obj, save_obj_with_mtl

class MeshGenerator:
    def __init__(self):
        this_py_dir = Path(__file__).parent.resolve()

        try:
            # Inject the cached binary into nvdiffrast to prevent recompiling
            _cached_plugin[False] = importlib.import_module("nvdiffrast_plugin")
        except:
            pass

        seed_everything(42)

        radius = 4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from Lrm import Lrm
        self.model = Lrm(self.device, encoder_freeze = False, encoder_model_name = "facebook/dino-vitb16", encoder_feat_dim = 768,
            transformer_dim = 1024, transformer_layers = 16, transformer_heads = 16, triplane_low_res = 32,
            triplane_high_res = 64, triplane_dim = 80, rendering_samples_per_ray = 128, grid_res = 128, grid_scale = 2.1, fovy = 30.0
        )

        model_ckpt_path = this_py_dir.joinpath(f"Models/instant_mesh_large.ckpt")
        if not model_ckpt_path.exists():
            print("Downloading pre-trained mesh generator models...")
            downloaded_model_ckpt_path = hf_hub_download(repo_id = "TencentARC/InstantMesh", filename = model_ckpt_path.name, repo_type = "model")
            shutil.copyfile(downloaded_model_ckpt_path, model_ckpt_path)

        state_dict = torch.load(model_ckpt_path, map_location = "cpu")["state_dict"]
        prefix = "lrm_generator."
        len_prefix = len(prefix)
        state_dict = {k[len_prefix : ] : v for k, v in state_dict.items() if k.startswith(prefix)}
        self.model.load_state_dict(state_dict, strict = True)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.input_cameras = get_zero123plus_input_cameras(batch_size = 1, radius = radius).to(self.device).squeeze(0)

    def Gen(self, images, texture_size, output_mesh_path : Path):
        mv_images = torch.empty(6, 3, 320, 320) # views, channels, height, width
        for i in range(0, 6):
            assert(images[i].size == (320, 320))
            mv_image = np.asarray(images[i], dtype = np.float32)
            mv_images[i] = torch.from_numpy(mv_image).permute(2, 0, 1).contiguous()

        mv_images = mv_images.to(self.device)
        mv_images /= 255.0
        mv_images = mv_images.clamp(0, 1)

        with torch.no_grad():
            vertices, faces, uvs, mesh_tex_idx, tex_map = self.model.GenerateMesh(mv_images, self.input_cameras, texture_size)
            save_obj_with_mtl(
                vertices.cpu().numpy(),
                uvs.cpu().numpy(),
                faces.cpu().numpy(),
                mesh_tex_idx.cpu().numpy(),
                tex_map.cpu().numpy(),
                output_mesh_path
            )
