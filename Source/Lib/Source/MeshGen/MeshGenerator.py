# Copyright (c) 2024 Minmin Gong
#

from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download
import numpy as np
from pytorch_lightning import seed_everything
import torch

from src.utils.camera_util import get_zero123plus_input_cameras

class MeshGenerator:
    def __init__(self):
        this_py_dir = Path(__file__).parent.resolve()

        seed_everything(42)

        radius = 4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from Lrm import Lrm
        self.model = Lrm(self.device, encoder_feat_dim = 768, transformer_dim = 1024, transformer_layers = 16, transformer_heads = 16,
            triplane_low_res = 32, triplane_high_res = 64, triplane_dim = 80, rendering_samples_per_ray = 128, grid_res = 128,
            grid_scale = 2.1, fovy = 30.0
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

    def GenPosMesh(self, images):
        mv_images = torch.empty(6, 3, 320, 320) # views, channels, height, width
        for i in range(0, 6):
            mv_image = np.frombuffer(images[i], dtype = np.uint8, count = 320 * 320 * 3)
            mv_image = torch.from_numpy(mv_image.copy()).to(self.device)
            mv_images[i] = mv_image.reshape(320, 320, 3).permute(2, 0, 1)

        mv_images = mv_images.float().contiguous()
        mv_images /= 255.0
        mv_images = mv_images.clamp(0, 1)

        with torch.no_grad():
            vertices, indices = self.model.GenerateMesh(mv_images, self.input_cameras)

        return (vertices.cpu().numpy().tobytes(), indices.cpu().numpy().tobytes())

    def QueryColors(self, positions, size):
        positions = np.frombuffer(positions, dtype = np.float32, count = size * 3)
        positions = torch.from_numpy(positions.copy()).to(self.device)
        positions = positions.reshape(-1, 3)

        with torch.no_grad():
            colors = self.model.QueryColors(positions)
            colors = torch.clamp(torch.round(colors * 255).int(), 0, 255).byte()
            colors = torch.cat([colors,
                         torch.full((colors.shape[0], 1), 255,
                             device = self.device, dtype = torch.uint8)],
                         dim = 1)

        return colors.cpu().numpy().tobytes()
