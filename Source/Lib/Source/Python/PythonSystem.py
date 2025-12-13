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

def TensorFromBytes(buffer : bytes, dtype : torch.dtype, count : int, device : Optional[torch.device] = None) -> torch.Tensor:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tensor = torch.frombuffer(buffer, dtype = dtype, count = count)
    if (device is None) or (device == GeneralDevice()):
        tensor = tensor.clone()
    else:
        tensor = tensor.to(device)
    return tensor

def TensorToBytes(tensor : torch.Tensor) -> bytes:
    tensor = tensor.to(GeneralDevice()).contiguous()
    return tensor.numpy().tobytes()
