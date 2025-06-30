# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/pipelines/samplers/classifier_free_guidance_mixin.py

class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def InferenceModel(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        pred = super().InferenceModel(model, x_t, t, cond, **kwargs)
        neg_pred = super().InferenceModel(model, x_t, t, neg_cond, **kwargs)
        return (1 + cfg_strength) * pred - cfg_strength * neg_pred
