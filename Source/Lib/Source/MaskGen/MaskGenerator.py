# Copyright (c) 2024 Minmin Gong
#

import numpy as np
from pathlib import Path

import onnxruntime as ort
from PIL import Image
from PIL.Image import Image as PILImage
import pooch

from cv2 import (
    BORDER_DEFAULT,
    MORPH_ELLIPSE,
    MORPH_OPEN,
    GaussianBlur,
    getStructuringElement,
    morphologyEx,
)

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

        self.device = "cuda"
        sess_opts = ort.SessionOptions()
        providers = ["CUDAExecutionProvider", ]
        self.inference_session = ort.InferenceSession(
            u2net_model_path,
            providers = providers,
            sess_options = sess_opts,
        )

        self.kernel = getStructuringElement(MORPH_ELLIPSE, (3, 3))

    def Gen(self, img):
        mask = self.Predict(img)
        mask = self.PostProcess(mask)
        return mask

    def Predict(self, img : PILImage) -> PILImage:
        norm_img = self.Normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (320, 320))
        norm_img_ort = ort.OrtValue.ortvalue_from_numpy(norm_img, self.device, 0)

        io_binding = self.inference_session.io_binding()
        io_binding.bind_input(name = self.inference_session.get_inputs()[0].name, device_type = self.device,
                              device_id = 0, element_type = np.float32, shape = norm_img_ort.shape(),
                              buffer_ptr = norm_img_ort.data_ptr())
        io_binding.bind_output(name = self.inference_session.get_outputs()[0].name, device_type = self.device)

        self.inference_session.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()

        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)

        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)

        mask = Image.fromarray((pred * 255).astype("uint8"), mode = "L")
        mask = mask.resize(img.size, Image.Resampling.LANCZOS)

        return mask

    def PostProcess(self, mask : PILImage) -> PILImage:
        mask = np.array(mask)
        mask = morphologyEx(mask, MORPH_OPEN, self.kernel)
        mask = GaussianBlur(mask, (5, 5), sigmaX = 2, sigmaY = 2, borderType = BORDER_DEFAULT)
        mask = np.where(mask < 127, 0, 255).astype(np.uint8)
        return Image.fromarray(mask)

    def Normalize(self, img, mean, std, size):
        img = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)

        img = np.array(img)
        img = img / np.max(img)

        norm_img = np.zeros((img.shape[0], img.shape[1], 3))
        norm_img[:, :, 0] = (img[:, :, 0] - mean[0]) / std[0]
        norm_img[:, :, 1] = (img[:, :, 1] - mean[1]) / std[1]
        norm_img[:, :, 2] = (img[:, :, 2] - mean[2]) / std[2]

        norm_img = norm_img.transpose((2, 0, 1))
        norm_img = np.expand_dims(norm_img, 0).astype(np.float32)

        return norm_img
