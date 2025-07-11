# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/pipelines/trellis_image_to_3d.py

from contextlib import contextmanager
import importlib
import json
from pathlib import Path
from typing import *
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision import transforms

from . import Samplers
from .. import Models
from ..Modules import Sparse as sp

from PythonSystem import GeneralDevice, WrapDinov2AttentionWithSdpa

class TrellisImageTo3DPipeline:
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler = None,
        sparse_structure_sampler_params = {},
        slat_sampler = None,
        slat_sampler_params = {},
        slat_normalization: dict = None,
        image_cond_model: str = None,
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
        backbones_module = getattr(importlib.import_module("dinov2.hub.backbones"), image_cond_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dinov2_model = backbones_module(pretrained = False)
        for i in range(len(dinov2_model.blocks)):
            dinov2_model.blocks[i].attn = WrapDinov2AttentionWithSdpa(dinov2_model.blocks[i].attn)

        compact_arch_name = "vitl"
        patch_size = 14
        num_register_tokens = 4
        model_full_name = f"dinov2_{compact_arch_name}{patch_size}_reg{num_register_tokens}"
        pth_file_name = this_py_dir.parents[1] / "Models/dinov2" / f"{model_full_name}_pretrain.pth"
        dinov2_model.load_state_dict(torch.load(pth_file_name, map_location = GeneralDevice(), weights_only = True))

        dinov2_model.eval()
        self.models["image_cond_model"] = dinov2_model
        self.image_cond_model_transform = transforms.Compose([
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def FromPretrained(path: str) -> "TrellisImageTo3DPipeline":
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
    def EncodeImage(self, images: torch.Tensor) -> torch.Tensor:
        assert images.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        images = self.image_cond_model_transform(images).to(self.device)
        features = self.models["image_cond_model"](images, is_training = True)["x_prenorm"]
        patch_tokens = functional.layer_norm(features, features.shape[-1 :])
        return patch_tokens

    def GetCond(self, images: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Get the conditioning information for the model.

        Args:
            images (torch.Tensor): The image prompts.

        Returns:
            cond: The conditional information.
            neg_cond: The netagive conditional information.
        """

        cond = self.EncodeImage(images)
        neg_cond = torch.zeros_like(cond)
        return cond, neg_cond

    def SampleSparseStructure(
        self,
        cond: torch.Tensor,
        neg_cond: torch.Tensor,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.

        Args:
            cond (torch.Tensor): The conditional information.
            neg_cond (torch.Tensor): The negative conditional information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """

        # Sample occupancy latent
        flow_model = self.models["sparse_structure_flow_model"]
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.Sample(
            flow_model,
            noise,
            cond,
            neg_cond,
            **sampler_params,
            verbose = True
        )

        # Decode occupancy latent
        decoder = self.models["sparse_structure_decoder"]
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()

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

        return self.models["slat_decoder_mesh"](slat)

    def SampleSlat(
        self,
        cond: torch.Tensor,
        neg_cond: torch.Tensor,
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
        flow_model = self.models["slat_flow_model"]
        noise = sp.SparseTensor(
            feats = torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords = coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.Sample(
            flow_model,
            noise,
            cond,
            neg_cond,
            **sampler_params,
            verbose = True
        )

        std = torch.tensor(self.slat_normalization["std"]).unsqueeze(0).to(slat.device)
        mean = torch.tensor(self.slat_normalization["mean"]).unsqueeze(0).to(slat.device)
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
        setattr(sampler, "old_inference_model", sampler.InferenceModel)

        if mode == "stochastic":
            if num_images > num_steps:
                print(f"Warning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def NewInferenceModel(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self.old_inference_model(model, x_t, t, cond = cond_i, **kwargs)
        elif mode == "multidiffusion":
            from .Samplers import FlowEulerSampler
            def NewInferenceModel(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler.InferenceModel(self, model, x_t, t, cond[i : i + 1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler.InferenceModel(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler.InferenceModel(self, model, x_t, t, cond[i : i + 1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        sampler.InferenceModel = NewInferenceModel.__get__(sampler, type(sampler))

        yield

        sampler.InferenceModel = sampler.old_inference_model
        delattr(sampler, "old_inference_model")

    @torch.no_grad()
    def Run(
        self,
        images: torch.Tensor,
        num_samples: int = 1,
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

        cond, neg_cond = self.GetCond(images)
        neg_cond = neg_cond[: 1]

        torch.manual_seed(1)

        num_images = images.shape[0]
        if num_images == 1:
            coords = self.SampleSparseStructure(cond, neg_cond, num_samples, sparse_structure_sampler_params)
            slat = self.SampleSlat(cond, neg_cond, coords, slat_sampler_params)
        else:
            ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get("steps")
            with self.InjectSamplerMultiImage("sparse_structure_sampler", num_images, ss_steps, mode = mode):
                coords = self.SampleSparseStructure(cond, neg_cond, num_samples, sparse_structure_sampler_params)
            slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get("steps")
            with self.InjectSamplerMultiImage("slat_sampler", num_images, slat_steps, mode = mode):
                slat = self.SampleSlat(cond, neg_cond, coords, slat_sampler_params)

        return self.DecodeSlat(slat)
