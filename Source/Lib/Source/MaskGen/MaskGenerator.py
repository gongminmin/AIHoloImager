# Copyright (c) 2024 Minmin Gong
#

from pathlib import Path

import numpy as np
import onnx
import onnx2torch
import requests
import torch

import Util

class MaskGenerator:
    def __init__(self):
        this_py_dir = Path(__file__).parent.resolve()
        u2net_model_path = this_py_dir.joinpath("Models/u2net.onnx")
        if not u2net_model_path.exists():
            print("Downloading pre-trained mask generator model...")
            Util.DownloadFile(f"https://github.com/danielgatis/rembg/releases/download/v0.0.0/{u2net_model_path.name}", u2net_model_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        onnx_model = onnx.load(u2net_model_path)
        self.u2net = onnx2torch.convert(onnx_model)
        self.u2net.eval()
        self.u2net.to(self.device)

    def Destroy(self):
        del self.u2net
        del self.device
        torch.cuda.empty_cache()

    @torch.no_grad()
    def Gen(self, img_data : bytes, width : int, height : int, num_channels : int) -> bytes:
        norm_img = np.frombuffer(img_data, dtype = np.float32, count = width * height * num_channels)
        norm_img = torch.from_numpy(norm_img.copy()).to(self.device)
        norm_img = norm_img.reshape(1, num_channels, height, width)

        outs = self.u2net(norm_img)
        pred = outs[0][:, 0, :, :]

        pred = pred.cpu().numpy()
        return pred.tobytes()
