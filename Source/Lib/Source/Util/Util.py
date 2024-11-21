# Copyright (c) 2024 Minmin Gong
#

def SeedRandom(seed : int):
    import random
    random.seed(seed)

    import numpy
    numpy.random.seed(seed)

    import torch
    torch.manual_seed(seed)

def DownloadFile(url, target_path, chunk_size = 8192):
    import requests

    with requests.get(url, stream = True) as response:
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(target_path, "wb") as file:
            for chunk in response.iter_content(chunk_size = chunk_size):
                file.write(chunk)
