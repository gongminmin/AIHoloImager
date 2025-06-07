# Copyright (c) 2025 Minmin Gong
#

from typing import *

import torch

def SeedRandom(seed : int):
    import random
    random.seed(seed)

    import numpy
    numpy.random.seed(seed)

    torch.manual_seed(seed)

compute_device_name = "cpu"
def InitPySys(device : str):
    global compute_device_name
    torch_device = getattr(torch, device, None)
    if (torch_device != None) and hasattr(torch_device, "is_available") and torch_device.is_available():
        compute_device_name = device

    print(f"Pick \"{compute_device_name}\" as the computation device.")

    import os
    os.environ["XFORMERS_DISABLED"] = "1" # Disable the usage of xformers in dinov2

    SeedRandom(42)

    if os.name == "nt":
        from pathlib import Path
        this_py_dir = Path(__file__).parent.resolve()

        # pywintypes imports modules in these directories
        import sys
        sys.path.append(str(this_py_dir / "Python/Lib/site-packages/win32"))
        sys.path.append(str(this_py_dir / "Python/Lib/site-packages/win32/lib"))

general_device = None
def GeneralDevice():
    global general_device
    if general_device == None:
        general_device = torch.device("cpu")
    return general_device

compute_device = None
def ComputeDevice():
    global compute_device
    if compute_device == None:
        global compute_device_name
        if compute_device_name != "cpu":
            compute_device = torch.device(compute_device_name)
        else:
            compute_device = GeneralDevice()
    return compute_device

def PurgeTorchCache():
    global compute_device_name
    torch_device = getattr(torch, compute_device_name, None)
    if (torch_device != None) and hasattr(torch_device, "empty_cache"):
        torch_device.empty_cache()

# From MoGe, https://github.com/microsoft/MoGe/blob/main/moge/model/utils.py
def WrapDinov2AttentionWithSdpa(module : torch.nn.Module):
    class AttentionWrapper(module.__class__):
        def forward(self, x : torch.Tensor, attn_bias : Optional[torch.Tensor] = None) -> torch.Tensor:
            batch, num, comp = x.shape
            qkv = self.qkv(x).reshape(batch, num, 3, self.num_heads, comp // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B, H, N, C // H)

            query, key, value = torch.unbind(qkv, 0)      # (B, H, N, C // H)

            x = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_bias)
            x = x.permute(0, 2, 1, 3).reshape(batch, num, comp) 

            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    module.__class__ = AttentionWrapper
    return module
