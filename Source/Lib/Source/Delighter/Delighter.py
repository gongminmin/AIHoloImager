# Copyright (c) 2025 Minmin Gong
#

from pathlib import Path

import numpy as np
import torch

class Delighter:
    def __init__(self):
        this_py_dir = Path(__file__).parent.resolve()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def Destroy(self):
        del self.device
        torch.cuda.empty_cache()

    @torch.no_grad()
    def Process(self, image, width, height, channels):
        image = np.frombuffer(image, dtype = np.uint8, count = height * width * channels)
        image = torch.from_numpy(image.copy()).to(self.device)
        image = image.reshape(height, width, channels)

        float_image = image.permute(2, 0, 1)
        float_image = float_image[0 : 3, :, :].float().contiguous()
        float_image /= 255.0

        float_image = float_image.unsqueeze(0)

        # Will be replaced by real delighting
        result_image = float_image

        result_image = result_image.squeeze(0)

        result_image = (result_image * 255).byte()
        result_image = result_image.permute(1, 2, 0)

        image[:, :, 0 : 3] = result_image

        return image.cpu().numpy().tobytes()
