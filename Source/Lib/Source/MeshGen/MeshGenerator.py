# Copyright (c) 2024 Minmin Gong
#

from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download
import numpy as np
from pytorch_lightning import seed_everything
import torch

from src.utils.camera_util import get_zero123plus_input_cameras
from Lrm import Lrm

class MeshGenerator:
    def __init__(self, grid_res : int, grid_scale : float):
        this_py_dir = Path(__file__).parent.resolve()

        seed_everything(42)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Lrm(self.device, encoder_feat_dim = 768, transformer_dim = 1024, transformer_layers = 16, transformer_heads = 16,
            triplane_low_res = 32, triplane_high_res = 64, triplane_dim = 80, grid_res = grid_res, grid_scale = grid_scale
        )

        model_ckpt_path = this_py_dir.joinpath(f"Models/instant_mesh_large.ckpt")
        if not model_ckpt_path.exists():
            print("Downloading pre-trained mesh generator models...")
            downloaded_model_ckpt_path = hf_hub_download(repo_id = "TencentARC/InstantMesh", filename = model_ckpt_path.name, repo_type = "model")
            shutil.copyfile(downloaded_model_ckpt_path, model_ckpt_path)

        loaded_state_dict = torch.load(model_ckpt_path, map_location = "cpu", weights_only = True)["state_dict"]

        self.decoder_density_nn = [None] * 4
        self.decoder_deformation_nn = [None] * 4
        self.decoder_color_nn = [None] * 4

        # Remove the prefix from state_dict since we don't have a lrm_generator class wrapping the model
        prefix = "lrm_generator."
        len_prefix = len(prefix)
        state_dict = {}
        for key, value in loaded_state_dict.items():
            if key.startswith(prefix):
                key = key[len_prefix : ]
                if key.startswith("synthesizer.decoder."):
                    key_tokens = key.split(".")
                    layer = int(key_tokens[3]) // 2
                    if key_tokens[4] == "weight":
                        index = 0
                    else:
                        index = 1
                    if key_tokens[2] == "net_sdf":
                        if index == 0:
                            self.decoder_density_nn[layer] = [None] * 2
                        self.decoder_density_nn[layer][index] = value
                    elif key_tokens[2] == "net_deformation":
                        if index == 0:
                            self.decoder_deformation_nn[layer] = [None] * 2
                        self.decoder_deformation_nn[layer][index] = value
                    elif key_tokens[2] == "net_rgb":
                        if index == 0:
                            self.decoder_color_nn[layer] = [None] * 2
                        self.decoder_color_nn[layer][index] = value
                else:
                    state_dict[key] = value
        self.model.load_state_dict(state_dict, strict = True)

        self.model = self.model.to(self.device)
        self.model.eval()

    def Destroy(self):
        del self.model
        del self.device
        torch.cuda.empty_cache()

    def GenNeRF(self, images):
        with torch.no_grad():
            mv_images = torch.empty(6, 3, 320, 320) # views, channels, height, width
            for i in range(0, 6):
                mv_image = np.frombuffer(images[i], dtype = np.uint8, count = 320 * 320 * 3)
                mv_image = torch.from_numpy(mv_image.copy()).to(self.device)
                mv_images[i] = mv_image.reshape(320, 320, 3).permute(2, 0, 1)

            mv_images = mv_images.float().contiguous()
            mv_images /= 255.0
            mv_images = mv_images.clamp(0, 1)

            input_cameras = get_zero123plus_input_cameras(batch_size = 1, radius = 4).to(self.device).squeeze(0)
            planes = self.model.GenNeRF(mv_images, input_cameras)

        return (planes.shape[0], planes.shape[1], planes.shape[2], planes.shape[3], planes.cpu().numpy().tobytes())

    def NumDensityNnLayers(self):
        return len(self.decoder_density_nn)

    def DensityNnSize(self, layer):
        size = self.decoder_density_nn[layer][0].shape
        return (size[0], size[1])

    def DensityNnWeight(self, layer):
        return self.decoder_density_nn[layer][0].cpu().numpy().tobytes()

    def DensityNnBias(self, layer):
        return self.decoder_density_nn[layer][1].cpu().numpy().tobytes()

    def NumDeformationNnLayers(self):
        return len(self.decoder_deformation_nn)

    def DeformationNnSize(self, layer):
        size = self.decoder_deformation_nn[layer][0].shape
        return (size[0], size[1])

    def DeformationNnWeight(self, layer):
        return self.decoder_deformation_nn[layer][0].cpu().numpy().tobytes()

    def DeformationNnBias(self, layer):
        return self.decoder_deformation_nn[layer][1].cpu().numpy().tobytes()

    def NumColorNnLayers(self):
        return len(self.decoder_color_nn)

    def ColorNnSize(self, layer):
        size = self.decoder_color_nn[layer][0].shape
        return (size[0], size[1])

    def ColorNnWeight(self, layer):
        return self.decoder_color_nn[layer][0].cpu().numpy().tobytes()

    def ColorNnBias(self, layer):
        return self.decoder_color_nn[layer][1].cpu().numpy().tobytes()
