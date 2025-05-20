# Copyright (c) 2025 Minmin Gong
#

def SeedRandom(seed : int):
    import random
    random.seed(seed)

    import numpy
    numpy.random.seed(seed)

    import torch
    torch.manual_seed(seed)

compute_on_cuda = False
def InitPySys(enable_cuda : bool):
    import torch

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
        import torch
        general_device = torch.device("cpu")
    return general_device

compute_device = None
def ComputeDevice():
    global compute_device
    if compute_device == None:
        global compute_on_cuda
        if compute_on_cuda:
            import torch
            compute_device = torch.device("cuda")
        else:
            compute_device = GeneralDevice()
    return compute_device

def PurgeTorchCache():
    global compute_device
    if (compute_device != None) and (compute_device.type == "cuda"):
        import torch
        torch.cuda.empty_cache()
