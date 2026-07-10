# Copyright (c) 2025-2026 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/pipelines/samplers/guidance_interval_mixin.py

import torch
import torch.nn as nn

class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def InferenceModel(self, model: nn.Module, x_t: torch.Tensor, t: float, cond: torch.Tensor, neg_cond: torch.Tensor, cfg_strength: float, cfg_interval: float, **kwargs):
        pred = super().InferenceModel(model, x_t, t, cond, **kwargs)
        if cfg_interval[0] <= t <= cfg_interval[1]:
            neg_pred = super().InferenceModel(model, x_t, t, neg_cond, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            return pred
