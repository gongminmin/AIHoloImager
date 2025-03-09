# Copyright (c) 2024-2025 Minmin Gong
#

def SeedRandom(seed : int):
    import random
    random.seed(seed)

    import numpy
    numpy.random.seed(seed)

    import torch
    torch.manual_seed(seed)
