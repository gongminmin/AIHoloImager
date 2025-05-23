# Copyright (c) 2025 Minmin Gong
#

import torch

def SeedRandom(seed : int):
    import random
    random.seed(seed)

    import numpy
    numpy.random.seed(seed)

    torch.manual_seed(seed)

compute_on_cuda = False
def InitPySys(enable_cuda : bool):
    global compute_on_cuda
    compute_on_cuda = enable_cuda and torch.cuda.is_available()

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
        global compute_on_cuda
        if compute_on_cuda:
            compute_device = torch.device("cuda")
        else:
            compute_device = GeneralDevice()
    return compute_device

def PurgeTorchCache():
    global compute_device
    if (compute_device != None) and (compute_device.type == "cuda"):
        torch.cuda.empty_cache()

# From MoGe, https://github.com/microsoft/MoGe/blob/main/moge/model/utils.py
def WrapDinov2AttentionWithSdpa(module : torch.nn.Module):
    class AttentionWrapper(module.__class__):
        def forward(self, x : torch.Tensor, attn_bias = None) -> torch.Tensor:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B, H, N, C // H)

            q, k, v = torch.unbind(qkv, 0)      # (B, H, N, C // H)

            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_bias)
            x = x.permute(0, 2, 1, 3).reshape(B, N, C) 

            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    module.__class__ = AttentionWrapper
    return module
