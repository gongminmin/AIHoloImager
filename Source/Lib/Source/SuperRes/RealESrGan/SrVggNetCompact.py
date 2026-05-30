# Copyright (c) 2026 Minmin Gong
#

from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch.nn.utils import skip_init

from PythonSystem import GeneralDevice

class SrVggNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.
    It is a compact network structure, which performs upsampling in the last layer and no convolution is conducted on the HR feature space.
    """

    def __init__(
        self,
        in_channels: Optional[int] = 3,
        out_channels: Optional[int] = 3,
        num_feat: Optional[int] = 64,
        num_conv: Optional[int] = 16,
        upscale: Optional[float] = 4,
        act_type: Optional[str] = "prelu",
        device : Optional[torch.device] = None
    ):
        super(SrVggNetCompact, self).__init__()

        body_layers = []
        for i in range(num_conv + 2):
            in_ch = in_channels if i == 0 else num_feat
            out_ch = num_feat if i < num_conv + 1 else out_channels * upscale * upscale
            body_layers.append(nn.Conv2d(in_ch, out_ch, 3, 1, 1, device = device))

            if i < num_conv + 1:
                if act_type == "relu":
                    activation = nn.ReLU(inplace = True)
                elif act_type == "prelu":
                    activation = nn.PReLU(num_parameters = num_feat)
                elif act_type == "leakyrelu":
                    activation = nn.LeakyReLU(negative_slope = 0.1, inplace = True)
                body_layers.append(activation)
        self.body = nn.Sequential(*body_layers)

    @staticmethod
    def FromPretrained(paths: List[Union[str, Path]], dni_weights: List[float]) -> "SrVggNetCompact":
        model = skip_init(SrVggNetCompact, in_channels = 3, out_channels = 3, num_feat = 64, num_conv = 32, upscale = 4, act_type = "prelu")

        load_net = SrVggNetCompact.DeepNNetworkInterpolation(paths, dni_weights)

        key_name = "params_ema" if "params_ema" in load_net else "params"
        model.load_state_dict(load_net[key_name], strict = True)

        model.eval()
        return model

    @staticmethod
    def DeepNNetworkInterpolation(paths: List[Union[str, Path]], dni_weights: List[float], key: Optional[str] = "params") -> dict:
        """
        Deep Network Interpolation for Continuous Imagery Effect Transition
        """

        assert len(paths) >= 2
        assert len(paths) == len(dni_weights)

        net_a = torch.load(paths[0], map_location = GeneralDevice(), weights_only = True)
        net_b = torch.load(paths[1], map_location = GeneralDevice(), weights_only = True)
        for k, v_a in net_a[key].items():
            net_a[key][k] = dni_weights[0] * v_a + dni_weights[1] * net_b[key][k]
        return net_a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.body(x)
        scale_sq = output.shape[1] // x.shape[1]
        output = output.reshape(output.shape[0], x.shape[1], scale_sq, output.shape[2], output.shape[3])
        return output
