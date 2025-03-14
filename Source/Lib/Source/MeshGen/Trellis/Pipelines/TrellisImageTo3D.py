# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/pipelines/trellis_image_to_3d.py

from contextlib import contextmanager
from pathlib import Path
import json
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision import transforms

from . import Samplers
from .. import Models
from ..Modules import Sparse as sp

class TrellisImageTo3DPipeline:
    def __init__(
        self,
        models : dict[str, nn.Module] = None,
        sparse_structure_sampler = None,
        sparse_structure_sampler_params = {},
        slat_sampler = None,
        slat_sampler_params = {},
        slat_normalization : dict = None,
        image_cond_model : str = None,
    ):
        assert models != None

        self.models = models
        for model in self.models.values():
            model.eval()

        self.sparse_structure_sampler = sparse_structure_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params

        self.slat_sampler = slat_sampler
        self.slat_sampler_params = slat_sampler_params
        self.slat_normalization = slat_normalization

        this_py_dir = Path(__file__).parent.resolve()
        dinov2_model = torch.hub.load(this_py_dir.parents[1] / "dinov2", image_cond_model, source = "local", pretrained = False)

        compact_arch_name = "vitl"
        patch_size = 14
        num_register_tokens = 4
        model_full_name = f"dinov2_{compact_arch_name}{patch_size}_reg{num_register_tokens}"
        pth_file_name = this_py_dir.parents[1] / "Models/dinov2" / f"{model_full_name}_pretrain.pth"
        dinov2_model.load_state_dict(torch.load(pth_file_name, map_location = "cpu", weights_only = True))

        dinov2_model.eval()
        self.models["image_cond_model"] = dinov2_model
        self.image_cond_model_transform = transforms.Compose([
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def FromPretrained(path : str) -> "TrellisImageTo3DPipeline":
        config_file_path = Path(path) / "pipeline.json"

        with open(config_file_path, "r") as file:
            args = json.load(file)["args"]

        models = {}
        for key, value in args["models"].items():
            if key not in ("slat_decoder_rf", "slat_decoder_gs"):
                models[key] = Models.FromPretrained(f"{path}/{value}")

        sparse_structure_sampler = getattr(Samplers, args["sparse_structure_sampler"]["name"])(**args["sparse_structure_sampler"]["args"])
        sparse_structure_sampler_params = args["sparse_structure_sampler"]["params"]
        slat_sampler = getattr(Samplers, args["slat_sampler"]["name"])(**args["slat_sampler"]["args"])
        slat_sampler_params = args["slat_sampler"]["params"]
        slat_normalization = args["slat_normalization"]
        image_cond_model = args["image_cond_model"]

        pipeline = TrellisImageTo3DPipeline(
            models,
            sparse_structure_sampler,
            sparse_structure_sampler_params,
            slat_sampler,
            slat_sampler_params,
            slat_normalization,
            image_cond_model
        )

        return pipeline

    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            if hasattr(model, "device"):
                return model.device
        for model in self.models.values():
            if hasattr(model, "parameters"):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    @torch.no_grad()
    def EncodeImage(self, images : torch.Tensor) -> torch.Tensor:
        assert images.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        images = self.image_cond_model_transform(images).to(self.device)
        features = self.models['image_cond_model'](images, is_training = True)["x_prenorm"]
        patch_tokens = functional.layer_norm(features, features.shape[-1 :])
        return patch_tokens

    def GetCond(self, images : torch.Tensor) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            images (torch.Tensor): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.EncodeImage(images)
        neg_cond = torch.zeros_like(cond)
        return {
            "cond" : cond,
            "neg_cond" : neg_cond,
        }

    def SampleSparseStructure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        )["samples"]
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()
    
        return coords

    def DecodeSlat(
        self,
        slat: sp.SparseTensor,
    ) -> list:
        """
        Decode the structured latent.
    
        Args:
            slat (sp.SparseTensor): The structured latent.
    
        Returns:
            list: A list of decoded structured latent.
        """
        ret = self.models["slat_decoder_mesh"](slat)
        return ret

    def SampleSlat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        )["samples"]
    
        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
    
    @contextmanager
    def InjectSamplerMultiImage(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal["stochastic", "multidiffusion"] = "stochastic",
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)
    
        if mode == "stochastic":
            if num_images > num_steps:
                print(f"Warning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.")
    
            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode == "multidiffusion":
            from .Samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))
    
        yield
    
        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')
    
    @torch.no_grad()
    def Run(
        self,
        images: torch.Tensor,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        mode: Literal["stochastic", "multidiffusion"] = "stochastic",
    ) -> list:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (torch.Tensor): The image or multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
        """
        cond = self.GetCond(images)
        cond["neg_cond"] = cond["neg_cond"][: 1]
        torch.manual_seed(seed)

        num_images = images.shape[0]
        if num_images == 1:
            coords = self.SampleSparseStructure(cond, num_samples, sparse_structure_sampler_params)
            slat = self.SampleSlat(cond, coords, slat_sampler_params)
        else:
            ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get("steps")
            with self.InjectSamplerMultiImage("sparse_structure_sampler", num_images, ss_steps, mode = mode):
                coords = self.SampleSparseStructure(cond, num_samples, sparse_structure_sampler_params)
            slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get("steps")
            with self.InjectSamplerMultiImage("slat_sampler", num_images, slat_steps, mode = mode):
                slat = self.SampleSlat(cond, coords, slat_sampler_params)

        return self.DecodeSlat(slat)
