# Copyright (c) 2024 Minmin Gong
#

from pathlib import Path

import numpy as np
import onnx
import onnx2torch
import pooch
import torch
import torchvision.transforms as transforms

class MaskGenerator:
    def __init__(self):
        this_py_dir = Path(__file__).parent.resolve()
        u2net_model_path = this_py_dir.joinpath("Models/u2net.onnx")
        if not u2net_model_path.exists():
            pooch.retrieve(
                "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
                None,
                fname = u2net_model_path.name,
                path = u2net_model_path.parent.resolve(),
                progressbar = True,
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        onnx_model = onnx.load(u2net_model_path)
        self.u2net = onnx2torch.convert(onnx_model)
        self.u2net.eval()
        self.u2net.to(self.device)

        self.kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype = torch.float32,
                                   device = self.device).unsqueeze(0).unsqueeze(0)

        self.blurer = transforms.GaussianBlur(kernel_size = (5, 5), sigma = 2.0).to(self.device)

    def Destroy(self):
        del self.u2net
        del self.kernel
        del self.blurer
        del self.device
        torch.cuda.empty_cache()

    def Gen(self, img_data : bytes, width : int, height : int, num_channels : int) -> bytes:
        with torch.no_grad():
            img = np.frombuffer(img_data, dtype = np.uint8, count = width * height * num_channels)
            img = torch.from_numpy(img.copy()).to(self.device)
            img = img.reshape(height, width, num_channels)

            mask = self.Predict(img)
            mask = self.PostProcess(mask)

            mask = mask.cpu().numpy()
        return mask.tobytes()

    def Predict(self, img : torch.Tensor) -> torch.Tensor:
        predict_size = (320, 320)
        norm_img = self.Normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), predict_size).contiguous()

        outs = self.u2net(norm_img)
        pred = outs[0][:, 0, :, :]
        pred = pred.contiguous()

        ma = torch.max(pred)
        mi = torch.min(pred)

        pred = (pred - mi) / (ma - mi)

        upscaler = transforms.Resize((img.shape[0], img.shape[1]), transforms.InterpolationMode.BICUBIC,
                                     antialias = True).to(self.device)
        mask = upscaler((pred * 255).byte())

        return mask # [C, H, W]

    def PostProcess(self, mask : torch.Tensor) -> torch.Tensor:
        mask = (mask.float() / 255.0).unsqueeze(0) # [B, C, H, W]

        mask = torch.gt(mask, 0.5).float()
        mask = 1 - torch.clamp(torch.nn.functional.conv2d(1 - mask, self.kernel, padding = (1, 1)), 0, 1)
        mask = torch.gt(mask, 0.5).float()
        mask = torch.clamp(torch.nn.functional.conv2d(mask, self.kernel, padding = (1, 1)), 0, 1)

        mask = mask.squeeze(0) # [C, H, W]
        mask = self.blurer((mask * 255).byte())
        mask = (torch.gt(mask, 127) * 255).byte()

        return mask # [C, H, W]

    def Normalize(self, img : torch.Tensor, mean, std, size) -> torch.Tensor:
        img = img[:, :, : 3]
        img = img.permute(2, 0, 1) # [C, H, W]
        downscaler = transforms.Resize((size[1], size[0]), transforms.InterpolationMode.BICUBIC,
                                       antialias = True).to(self.device)
        img = downscaler(img)

        img = img / torch.max(img)

        normalizer = transforms.Normalize(mean, std)
        norm_img = normalizer(img)
        norm_img = norm_img.unsqueeze(0).float()

        return norm_img
