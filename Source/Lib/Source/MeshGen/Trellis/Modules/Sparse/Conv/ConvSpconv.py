# Copyright (c) 2025 Minmin Gong
#

# Based on https://github.com/microsoft/TRELLIS/blob/main/trellis/modules/sparse/conv/conv_spconv.py

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .. import SparseTensor

from AIHoloImagerSubMConv import SubMConv3DHelper

subm_helpers = {}

class SparseConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int, int]], bias: Optional[bool] = True, indices_key: Optional[str] = None, device: Optional[torch.device] = None):
        super(SparseConv3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.indices_key = indices_key

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            assert isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 3, "kernel_size must be an int or a tuple/list of three ints"
            self.kernel_size = tuple(kernel_size)

        self.weight = nn.Parameter(torch.zeros((out_channels, *self.kernel_size, in_channels), device = device))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, device = device))
        else:
            self.register_parameter("bias", None)

    def SetGpuSystem(self, gpu_system):
        self.gpu_system = gpu_system

    def forward(self, sp_tensor: SparseTensor) -> SparseTensor:
        dtype = sp_tensor.dtype
        device = sp_tensor.device
        weight = self.weight.permute(1, 2, 3, 4, 0).contiguous().to(dtype = dtype, device = device)
        sp_values =  sp_tensor.feats

        if self.kernel_size == (1, 1, 1):
            curr_weight = weight[0, 0, 0].permute(1, 0).reshape(self.in_channels, self.out_channels)
            features_out = torch.mm(sp_values, curr_weight)
        else:
            sp_coords = sp_tensor.coords.to(device = device)

            global subm_helpers
            subm_helper = subm_helpers.get(self.indices_key, None)
            build_coords = False
            if subm_helper is None:
                subm_helper = SubMConv3DHelper(self.gpu_system, device)
                subm_helpers[self.indices_key] = subm_helper
                build_coords = True
            elif self.indices_key is None:
                build_coords = True
            if build_coords:
                subm_helper.BuildCoordsMap(sp_coords)

            base_z = -(self.kernel_size[0] // 2)
            base_y = -(self.kernel_size[1] // 2)
            base_x = -(self.kernel_size[2] // 2)

            curr_weight = weight[-base_z, -base_y, -base_x]
            features_out = torch.mm(sp_values, curr_weight)

            offsets = []
            for dz in range(self.kernel_size[0]):
                z = base_z + dz
                for dy in range(self.kernel_size[1]):
                    y = base_y + dy
                    for dx in range(self.kernel_size[2]):
                        x = base_x + dx
                        if (x, y, z) != (0, 0, 0):
                            offsets.append((dz, dy, dx))

            nei_indices, nei_indices_size = subm_helper.FindAvailableNeighbors((base_z, base_y, base_x), offsets);

            for i, offset in enumerate(offsets):
                nei = nei_indices[i, 0 : nei_indices_size[i]]
                if nei.shape[0] != 0:
                    dz, dy, dx = offset
                    curr_weight = weight[dz, dy, dx]
                    features_out[nei[:, 0]] += torch.mm(sp_values[nei[:, 1]], curr_weight)

        if self.bias is not None:
            features_out += self.bias.to(device = device)

        new_shape = (sp_tensor.shape[0], self.out_channels)

        return SparseTensor(
            features_out,
            sp_tensor.coords,
            shape = torch.Size(new_shape),
            layout = sp_tensor.layout,
            scale = sp_tensor._scale,
            spatial_cache = sp_tensor._spatial_cache,
        )

    def __call__(self, sp_tensor):
        return self.forward(sp_tensor)
