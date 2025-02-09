# Copyright (c) 2024 Minmin Gong
#

import math
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.resolve() / "MoGe"))

import numpy as np
import torch
from scipy import ndimage
from scipy.sparse import csr_matrix, linalg

from moge.model import MoGeModel

import Util

def WritePfm(file_path, data):
    if data.shape[2] == 1:
        identifier = "Pf"
    else:
        identifier = "PF"
    with open(file_path, "wb") as pfm:
        pfm.write(f"{identifier}\n{data.shape[1]} {data.shape[0]}\n-1.0\n".encode("utf-8"))
        pfm.write(np.flip(data, axis = 0).tobytes())

class Image2Depth:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        this_py_dir = Path(__file__).parent.resolve()
        moge_pt_path = this_py_dir.joinpath("Models/MoGe/model.pt")
        if not moge_pt_path.exists():
            print("Downloading pre-trained MoGe model...")
            Util.DownloadFile(f"https://huggingface.co/Ruicheng/moge-vitl/resolve/main/{moge_pt_path.name}", moge_pt_path)

        self.model = MoGeModel.from_pretrained(moge_pt_path)
        self.model.eval()
        self.model = self.model.to(self.device)

    def Destroy(self):
        del self.model
        del self.device
        torch.cuda.empty_cache()

    @torch.no_grad()
    def MoGeGenDepth(self, image, fov_x):
        return self.model.infer(image, fov_x = fov_x)["depth"]

    def MergeDepth(self, sfm_depth, sfm_mask, moge_depth):
        height, width = sfm_depth.shape[: 2]

        mask = sfm_mask
        mask_flat = mask.flatten()

        depth_scale_list = []
        depth_scale = np.zeros(moge_depth.shape, dtype = np.float32)
        for y in range(height):
            for x in range(width):
                if mask[y, x]:
                    depth_scale[y, x] = sfm_depth[y, x] / moge_depth[y, x]
                    depth_scale_list.append(depth_scale[y, x])

        depth_scale_list = np.array(depth_scale_list)
        num = len(depth_scale_list)
        depth_scale_list = np.sort(depth_scale_list)
        depth_scale_list = depth_scale_list[num // 5 : num - (num // 5)]
        avg_depth_scale = np.mean(depth_scale_list)

        gx = np.zeros((height, width), dtype = np.float32)
        for y in range(0, height):
            for x in range(1, width):
                gx[y, x] = (moge_depth[y, x] - moge_depth[y, x - 1]) * avg_depth_scale
        gy = np.zeros((height, width), dtype = np.float32)
        for y in range(1, height):
            for x in range(0, width):
                gy[y, x] = (moge_depth[y, x] - moge_depth[y - 1, x]) * avg_depth_scale

        laplacian_data = np.array([[-4, 1, 1, 1, 1]], dtype = np.float32).repeat(height * width, axis = 0)
        laplacian_data[mask_flat] = [1, 0, 0, 0, 0]
        laplacian_data = laplacian_data.reshape(-1)

        laplacian_indices = np.stack([
                np.arange(height * width),             # center
                np.arange(height * width) - width,     # up
                np.arange(height * width) + width,     # down
                np.arange(height * width) - 1,         # left
                np.arange(height * width) + 1          # right
            ], axis = -1)

        laplacian_indices[mask_flat, 1 :] = np.repeat(np.arange(height * width)[mask_flat, np.newaxis], 4, axis = 1)
        laplacian_indices[:, 1][laplacian_indices[:, 1] < 0] = np.arange(height * width)[laplacian_indices[:, 1] < 0]
        laplacian_indices[:, 2][laplacian_indices[:, 2] >= height * width] = np.arange(height * width)[laplacian_indices[:, 2] >= height * width]
        laplacian_indices[:, 3][np.arange(height * width) % width == 0] = np.arange(height * width)[np.arange(height * width) % width == 0]
        laplacian_indices[:, 4][(np.arange(height * width) + 1) % width == 0] = np.arange(height * width)[(np.arange(height * width) + 1) % width == 0]

        laplacian_indices = laplacian_indices.reshape(-1)
        laplacian_ind_ptr = np.arange(0, height * width * 5 + 1, 5)
        laplacian = csr_matrix((laplacian_data, laplacian_indices, laplacian_ind_ptr), shape = (height * width, height * width))

        b = np.zeros(height * width, dtype = np.float32)
        b[mask_flat] = sfm_depth.flatten()[mask_flat]
        for y in range(0, height - 1):
            for x in range(0, width - 1):
                index = y * width + x
                if not mask[y, x]:
                    b[index] = gx[y, x + 1] - gx[y, x] + gy[y + 1, x] - gy[y, x]

        merged_depth = linalg.spsolve(laplacian, b)
        merged_depth = merged_depth.astype(np.float32)
        merged_depth = merged_depth.reshape((height, width))
        return merged_depth

    @torch.no_grad()
    def Process(self, image, image_width, image_height, channels, fov_x, sfm_depth, sfm_confidence, depth_width, depth_height, roi):
        image = np.frombuffer(image, dtype = np.uint8, count = image_height * image_width * channels)
        image = torch.from_numpy(image.copy()).to(self.device)
        image = image.reshape(image_height, image_width, channels)[:, :, 0 : 3].permute(2, 0, 1)

        image = image.float().contiguous()
        image /= 255.0

        moge_depth = self.MoGeGenDepth(image, fov_x).cpu().numpy()
        cropped_moge_depth = moge_depth[roi[1] : roi[3], roi[0] : roi[2]]

        sfm_depth = np.frombuffer(sfm_depth, dtype = np.float32, count = depth_height * depth_width)
        sfm_depth = sfm_depth.reshape(depth_height, depth_width)

        sfm_confidence = np.frombuffer(sfm_confidence, dtype = np.uint8, count = depth_height * depth_width)
        sfm_confidence = sfm_confidence.reshape(depth_height, depth_width)

        sfm_mask = sfm_confidence >= 128
        #kernel = np.ones((3, 3), np.uint8)
        #sfm_mask = ndimage.binary_erosion(sfm_mask, structure = kernel)

        scale_x = image_width // depth_width
        scale_y = image_height // depth_height

        cropped_sfm_depth = np.zeros(cropped_moge_depth.shape, dtype = sfm_depth.dtype)
        cropped_sfm_mask = np.zeros(cropped_moge_depth.shape, dtype = sfm_mask.dtype)
        for y in range(0, cropped_moge_depth.shape[0], scale_y):
            for x in range(0, cropped_moge_depth.shape[1], scale_x):
                cropped_sfm_depth[y, x] = sfm_depth[(roi[1] + y) // scale_y, (roi[0] + x) // scale_x]
                cropped_sfm_mask[y, x] = sfm_mask[(roi[1] + y) // scale_y, (roi[0] + x) // scale_x]

        merged_depth = self.MergeDepth(cropped_sfm_depth, cropped_sfm_mask, cropped_moge_depth)

        return merged_depth.tobytes()
