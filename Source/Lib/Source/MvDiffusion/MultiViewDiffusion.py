# Copyright (c) 2024 Minmin Gong
#

from pathlib import Path

from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import numpy as np
from PIL import Image
import torch

import Util

class MultiViewDiffusion:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Util.SeedRandom(42)

        this_py_dir = Path(__file__).parent.resolve()
        pipeline_path = this_py_dir.joinpath("InstantMesh/zero123plus")

        pretrained_dir = this_py_dir.joinpath("Models/zero123plus-v1.2")
        reload_from_network = True
        if pretrained_dir.exists():
            print(f"Load from {pretrained_dir}.")
            try:
                self.pipeline = DiffusionPipeline.from_pretrained(
                    pretrained_dir,
                    custom_pipeline = str(pipeline_path),
                    torch_dtype = torch.float16,
                )
                reload_from_network = False
            except:
                print(f"Failed. Retry from network.")

        if reload_from_network:
            self.pipeline = DiffusionPipeline.from_pretrained(
                "sudo-ai/zero123plus-v1.2",
                custom_pipeline = str(pipeline_path),
                torch_dtype = torch.float16,
            )

            pretrained_dir.mkdir(parents = True, exist_ok = True)
            self.pipeline.save_pretrained(pretrained_dir)

        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config, timestep_spacing = "trailing"
        )

        unet_ckpt_path = this_py_dir.joinpath("Models/diffusion_pytorch_model.bin")
        if not unet_ckpt_path.exists():
            print("Downloading pre-trained multi-view diffusion model...")
            Util.DownloadFile(f"https://huggingface.co/TencentARC/InstantMesh/resolve/main/{unet_ckpt_path.name}", unet_ckpt_path);

        state_dict = torch.load(unet_ckpt_path, map_location = "cpu", weights_only = True)
        self.pipeline.unet.load_state_dict(state_dict, strict = True)

        self.pipeline = self.pipeline.to(device)

    def Destroy(self):
        del self.pipeline
        torch.cuda.empty_cache()

    @torch.no_grad()
    def Gen(self, input_image_data : bytes, width : int, height : int, num_channels : int, num_steps : int):
        if num_channels == 1:
            mode = "L"
        elif num_channels == 3:
            mode = "RGB"
        else:
            assert(num_channels == 4)
            mode = "RGBA"
        input_image = Image.frombuffer(mode, (width, height), input_image_data)
        image = np.array(self.pipeline(input_image, num_inference_steps = num_steps).images[0])
        return (image.tobytes(), image.shape[1], image.shape[0], image.shape[2])
