# Copyright (c) 2024 Minmin Gong
#

from pathlib import Path
import shutil

from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download
from pytorch_lightning import seed_everything
import torch

class MultiViewDiffusion:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed_everything(42)

        this_py_dir = Path(__file__).parent.resolve()
        pipeline_path = this_py_dir.joinpath("InstantMesh/zero123plus")

        self.pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2", 
            custom_pipeline = str(pipeline_path),
            torch_dtype = torch.float16,
        )

        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config, timestep_spacing = "trailing"
        )

        unet_ckpt_path = this_py_dir.joinpath("Models/diffusion_pytorch_model.bin")
        if not unet_ckpt_path.exists():
            print("Downloading pre-trained multi-view diffusion models...")
            downloaded_unet_ckpt_path = hf_hub_download(repo_id = "TencentARC/InstantMesh", filename = unet_ckpt_path.name, repo_type = "model")
            shutil.copyfile(downloaded_unet_ckpt_path, unet_ckpt_path)

        state_dict = torch.load(unet_ckpt_path, map_location = "cpu")
        self.pipeline.unet.load_state_dict(state_dict, strict = True)

        self.pipeline = self.pipeline.to(device)

    def Gen(self, input_image, num_steps : int):
        return self.pipeline(input_image, num_inference_steps = num_steps).images[0]
