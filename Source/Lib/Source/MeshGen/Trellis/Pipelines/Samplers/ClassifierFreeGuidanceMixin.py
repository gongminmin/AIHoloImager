# Copyright (c) 2025-2026 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/pipelines/samplers/classifier_free_guidance_mixin.py

import torch
import torch.nn as nn

class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def InferenceModel(self, model: nn.Module, x_t: torch.Tensor, t: float, cond: torch.Tensor, neg_cond: torch.Tensor, cfg_strength: float, **kwargs) -> torch.Tensor:
        pred = super().InferenceModel(model, x_t, t, cond, **kwargs)
        neg_pred = super().InferenceModel(model, x_t, t, neg_cond, **kwargs)
        return (1 + cfg_strength) * pred - cfg_strength * neg_pred
