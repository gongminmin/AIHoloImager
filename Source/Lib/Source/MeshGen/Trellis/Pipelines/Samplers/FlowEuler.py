# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/pipelines/samplers/flow_euler.py

from typing import *

import numpy as np
import torch
from tqdm import tqdm

from .ClassifierFreeGuidanceMixin import ClassifierFreeGuidanceSamplerMixin
from .GuidanceIntervalMixin import GuidanceIntervalSamplerMixin

class FlowEulerSampler:
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """

    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def VToXStartEps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def InferenceModel(self, model, x_t, t, cond = None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device = x_t.device, dtype = torch.float32)
        return model(x_t, t, cond, **kwargs)

    def GetModelPrediction(self, model, x_t, t, cond = None, **kwargs):
        pred_v = self.InferenceModel(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self.VToXStartEps(x_t = x_t, t = t, v = pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def SampleOnce(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """

        pred_x_0, pred_eps, pred_v = self.GetModelPrediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return {"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0}

    @torch.no_grad()
    def Sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """

        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = {"samples": None, "pred_x_t": [], "pred_x_0": []}
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.SampleOnce(model, sample, t, t_prev, cond, **kwargs)
            sample = out["pred_x_prev"]
            ret["pred_x_t"].append(out["pred_x_prev"])
            ret["pred_x_0"].append(out["pred_x_0"])
        ret["samples"] = sample
        return ret

class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def Sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """

        return super().Sample(model, noise, cond, steps, rescale_t, verbose, neg_cond = neg_cond, cfg_strength = cfg_strength, **kwargs)

class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def Sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """

        return super().Sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength = cfg_strength, cfg_interval = cfg_interval, **kwargs)
