# Copyright (c) 2025 Minmin Gong
#

from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import skip_init

from PythonSystem import ComputeDevice, GeneralDevice, PurgeTorchCache
from ModMidas.MidasNet import MidasNet, MidasNetSmall

def Round32(x):
    return (int(x) + 31) & ~31

def SrgbToLinear(img):
    return img ** 2.2

def LinearToSrgb(img):
    img = img ** (1 / 2.2)
    return img.clip(0, 1)

def Invert(x):
    out = 1.0 / (x + 1.0)
    return out

def Uninvert(x):
    out = (1.0 / x) - 1.0
    return out

def GetYFactor():
    return (0.299, 0.587, 0.114)

def CalcLum(rgb):
    r = rgb[:, 0, :, :]
    g = rgb[:, 1, :, :]
    b = rgb[:, 2, :, :]

    yuv_factor = GetYFactor()
    l = r * yuv_factor[0] + g * yuv_factor[1] + b * yuv_factor[2]
    return l

def Rgb2InvLuv(rgb, eps = 0.001):
    l = CalcLum(rgb)

    r = rgb[:, 0, :, :]
    g = rgb[:, 1, :, :].clip(eps)
    b = rgb[:, 2, :, :]

    inv_l = Invert(l)
    inv_u = Invert(r / g)
    inv_v = Invert(b / g)

    return torch.stack((inv_l, inv_u, inv_v), axis = 1)

def InvLuv2Rgb(inv_luv, eps = 0.001):
    l = Uninvert(inv_luv[:, 0, :, :].clip(eps))
    u = Uninvert(inv_luv[:, 1, :, :].clip(eps))
    v = Uninvert(inv_luv[:, 2, :, :].clip(eps))

    yuv_factor = GetYFactor()
    g = l / (u * yuv_factor[0] + v * yuv_factor[2] + yuv_factor[1])
    r = g * u
    b = g * v

    return torch.stack((r, g, b), axis = 1)

def BaseResize(img, base_dim):
    h, w = img.shape[-2 :]

    max_dim = max(w, h)
    scale = base_dim / max_dim

    new_w = Round32(w * scale)
    new_h = Round32(h * scale)

    return torch.nn.functional.interpolate(img, size = (new_h, new_w), mode = "bilinear", align_corners = True, antialias = True)

def EqualizePredictions(img, base, full, p = 0.5):
    h, w = img.shape[-2 :]

    full_shading = Uninvert(full.clip(1e-5))
    base_shading = Uninvert(base.clip(1e-5))

    lum = CalcLum(img)
    
    full_albedo = lum / full_shading.clip(1e-5)
    base_albedo = lum / base_shading.clip(1e-5)

    rand_msk = (torch.randn(h, w) > p).unsqueeze(0).unsqueeze(0)

    flat_full_albedo = full_albedo[rand_msk]
    flat_base_albedo = base_albedo[rand_msk]

    scale = torch.linalg.lstsq(flat_full_albedo.reshape(-1, 1), flat_base_albedo, rcond = None)[0]

    new_full_albedo = scale * full_albedo
    new_full_shading = lum / new_full_albedo.clip(1e-5)
    new_full = Invert(new_full_shading)

    return base, new_full

class Delighter:
    def __init__(self):
        self.ord_model_idx = 0
        self.iid_model_idx = 1
        self.col_model_idx = 2
        self.alb_model_idx = 3

        this_py_dir = Path(__file__).parent.resolve()

        self.device = ComputeDevice()

        model_paths = (
            this_py_dir / "Models/Intrinsic/stage_0.pt",
            this_py_dir / "Models/Intrinsic/stage_1.pt",
            this_py_dir / "Models/Intrinsic/stage_2.pt",
            this_py_dir / "Models/Intrinsic/stage_3.pt",
        )
        self.LoadModels(model_paths)

    def Destroy(self):
        del self.models
        PurgeTorchCache()

    @torch.no_grad()
    def Process(self, image, width, height, channels):
        image = np.frombuffer(image, dtype = np.uint8, count = height * width * channels)
        image = torch.from_numpy(image.copy()).to(self.device)
        image = image.reshape(height, width, channels)

        float_image = image.permute(2, 0, 1)
        float_image = float_image[0 : 3, :, :].float().contiguous()
        float_image /= 255.0

        float_image = float_image.unsqueeze(0)

        orig_h, orig_w = float_image.shape[-2 :]
        base_dim = 384

        img, inv_gray_shading, gray_albedo = self.RunGrayPipeline(float_image, base_dim)

        rounded_size = img.shape[-2 :]
        scale = base_dim / max(rounded_size)
        base_size = (Round32(rounded_size[0] * scale), Round32(rounded_size[1] * scale))

        inv_img_luv = Rgb2InvLuv(img)
        inv_albedo_luv = Rgb2InvLuv(gray_albedo)

        inv_img_luv = torch.nn.functional.interpolate(inv_img_luv, size = base_size, mode = "bilinear", align_corners = True, antialias = True)
        inv_albedo_luv = torch.nn.functional.interpolate(inv_albedo_luv, size = base_size, mode = "bilinear", align_corners = True, antialias = True)
        inv_base_gray_shading = torch.nn.functional.interpolate(inv_gray_shading, size = base_size, mode = "bilinear", align_corners = True, antialias = True)

        combined = torch.cat([inv_img_luv, inv_base_gray_shading, inv_albedo_luv], 1)
        inv_uv_shading = self.models[self.col_model_idx](combined)

        inv_uv_shading = torch.nn.functional.interpolate(inv_uv_shading, size = rounded_size, mode = "bilinear", align_corners = True, antialias = True)

        inv_luv_shading = torch.cat((inv_gray_shading, inv_uv_shading), 1)
        rough_shading = InvLuv2Rgb(inv_luv_shading)
        rough_albedo = img / rough_shading

        rough_albedo *= 0.75 / torch.quantile(rough_albedo, 0.99)
        rough_albedo = rough_albedo.clip(1e-3)
        inv_rough_shading = Invert(img / rough_albedo)

        combined = torch.cat([img, inv_rough_shading, rough_albedo], 1)
        pred_albedo = self.models[self.alb_model_idx](combined)

        high_res_albedo = torch.nn.functional.interpolate(pred_albedo, size = (orig_h, orig_w), mode = "bilinear", align_corners = True, antialias = True)
        result_image = LinearToSrgb(high_res_albedo)

        result_image = result_image.squeeze(0)

        result_image = (result_image * 255).byte()
        result_image = result_image.permute(1, 2, 0)

        return result_image.cpu().numpy().tobytes()

    def LoadModels(self, paths):
        self.models = [None] * 4

        ord_state_dict = self.FixStateDict(torch.load(paths[0], map_location = GeneralDevice(), weights_only = True))
        iid_state_dict = self.FixStateDict(torch.load(paths[1], map_location = GeneralDevice(), weights_only = True))
        col_state_dict = self.FixStateDict(torch.load(paths[2], map_location = GeneralDevice(), weights_only = True))
        alb_state_dict = self.FixStateDict(torch.load(paths[3], map_location = GeneralDevice(), weights_only = True))

        ord_model = skip_init(MidasNet)
        ord_model.load_state_dict(ord_state_dict)
        ord_model.eval()
        self.models[self.ord_model_idx] = ord_model.to(self.device)

        iid_model = skip_init(MidasNetSmall, in_channels = 5, out_channels = 1)
        iid_model.load_state_dict(iid_state_dict)
        iid_model.eval()
        self.models[self.iid_model_idx] = iid_model.to(self.device)

        col_model = skip_init(MidasNet, activation = "sigmoid", in_channels = 7, out_channels = 2)
        col_model.load_state_dict(col_state_dict)
        col_model.eval()
        self.models[self.col_model_idx] = col_model.to(self.device)

        alb_model = skip_init(MidasNet, activation = "sigmoid", in_channels = 9, out_channels = 3)
        alb_model.load_state_dict(alb_state_dict)
        alb_model.eval()
        self.models[self.alb_model_idx] = alb_model.to(self.device)

    def FixStateDict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.find("resConfUnit") != -1:
                key = key.replace("resConfUnit", "res_conv_unit")
            new_state_dict[key] = value
        return new_state_dict

    @torch.no_grad()
    def RunGrayPipeline(self, img, base_dim, lstsq_p = 0.0):
        orig_h, orig_w = img.shape[-2 :]

        img = torch.nn.functional.interpolate(img, size = (Round32(orig_h), Round32(orig_w)), mode = "bilinear", align_corners = True, antialias = True)
        fh, fw = img.shape[-2 :]

        linear_img = SrgbToLinear(img)

        # Ordinal shading estimation

        base_input = BaseResize(linear_img, base_dim)
        full_input = linear_img

        base_out = self.models[self.ord_model_idx](base_input)
        base_out = torch.nn.functional.interpolate(base_out, size = (fh, fw), mode = "bilinear", align_corners = True, antialias = True)

        full_out = self.models[self.ord_model_idx](full_input)

        ordinal_base, ordinal_full = EqualizePredictions(linear_img, base_out, full_out, p = lstsq_p)

        # Ordinal shading to real shading

        combined = torch.cat((linear_img, ordinal_base, ordinal_full), 1)
        inv_shading = self.models[self.iid_model_idx](combined)
        
        shading = Uninvert(inv_shading.clip(1e-3))
        albedo = linear_img / shading

        return linear_img, inv_shading, albedo
