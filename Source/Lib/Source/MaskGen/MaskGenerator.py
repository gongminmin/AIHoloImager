# Copyright (c) 2024 Minmin Gong
#

from pathlib import Path

import numpy as np
import onnx
import onnx2torch
import requests
import torch

class MaskGenerator:
    def __init__(self):
        this_py_dir = Path(__file__).parent.resolve()
        u2net_model_path = this_py_dir.joinpath("Models/u2net.onnx")
        if not u2net_model_path.exists():
            response = requests.get("https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx")
            with open(u2net_model_path, "wb") as file:
                file.write(response.content)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        onnx_model = onnx.load(u2net_model_path)
        self.u2net = onnx2torch.convert(onnx_model)
        self.u2net.eval()
        self.u2net.to(self.device)

    def Destroy(self):
        del self.u2net
        del self.device
        torch.cuda.empty_cache()

    def Gen(self, img_data : bytes, width : int, height : int, num_channels : int) -> bytes:
        with torch.no_grad():
            norm_img = np.frombuffer(img_data, dtype = np.float32, count = width * height * num_channels)
            norm_img = torch.from_numpy(norm_img.copy()).to(self.device)
            norm_img = norm_img.reshape(1, num_channels, height, width)

            outs = self.u2net(norm_img)
            pred = outs[0][:, 0, :, :]

            pred = pred.cpu().numpy()
        return pred.tobytes()
