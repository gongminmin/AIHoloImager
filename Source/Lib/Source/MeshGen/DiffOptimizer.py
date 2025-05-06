# Copyright (c) 2024 Minmin Gong
#

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

from AIHoloImagerGpuDiffRender import GpuDiffRenderTorch, Viewport

def ScaleMatrix(scale):
    device = scale.device
    zero = torch.zeros(1, dtype = torch.float32, device = device).squeeze(0)
    r0 = torch.stack([scale[0], zero.detach().clone(), zero.detach().clone(), zero.detach().clone()], axis = 0)
    r1 = torch.stack([zero.detach().clone(), scale[1], zero.detach().clone(), zero.detach().clone()], axis = 0)
    r2 = torch.stack([zero.detach().clone(), zero.detach().clone(), scale[2], zero.detach().clone()], axis = 0)
    r3 = torch.tensor([0, 0, 0, 1], dtype = torch.float32, device = device)
    ret = torch.stack([r0, r1, r2, r3])
    return ret

def TranslateMatrix(translation):
    device = translation.device
    r0 = torch.tensor([1, 0, 0, 0], dtype = torch.float32, device = device)
    r1 = torch.tensor([0, 1, 0, 0], dtype = torch.float32, device = device)
    r2 = torch.tensor([0, 0, 1, 0], dtype = torch.float32, device = device)
    r3 = torch.cat([translation, torch.ones(1, device = device)], axis = 0)
    ret = torch.stack([r0, r1, r2, r3])
    return ret

def RotationMatrix(rot):
    device = rot.device
    zero = torch.zeros(1, dtype = torch.float32, device = device).squeeze(0)
    r0 = torch.stack([1 - 2 * rot[1] ** 2 - 2 * rot[2] ** 2, 2 * rot[0] * rot[1] + 2 * rot[2] * rot[3], 2 * rot[0] * rot[2] - 2 * rot[1] * rot[3], zero.detach().clone()])
    r1 = torch.stack([2 * rot[0] * rot[1] - 2 * rot[2] * rot[3], 1 - 2 * rot[0] ** 2 - 2 * rot[2] ** 2, 2 * rot[1] * rot[2] + 2 * rot[0] * rot[3], zero.detach().clone()])
    r2 = torch.stack([2 * rot[0] * rot[2] + 2 * rot[1] * rot[3], 2 * rot[1] * rot[2] - 2 * rot[0] * rot[3], 1 - 2 * rot[0] ** 2 - 2 * rot[1] ** 2, zero.detach().clone()])
    r3 = torch.tensor([0, 0, 0, 1], dtype = torch.float32, device = device)
    ret = torch.stack([r0, r1, r2, r3])
    return ret

def ComposeMatrix(scale, rotation, translation):
    scale_mtx = ScaleMatrix(scale)
    rotation_mtx = RotationMatrix(rotation)
    translation_mtx = TranslateMatrix(translation)
    return torch.matmul(torch.matmul(scale_mtx, rotation_mtx), translation_mtx)

def NormalizeQuat(q):
    return q / torch.sum(q ** 2) ** 0.5

class DiffOptimizer:
    def __init__(self, gpu_system):
        self.downsampling = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_dr = GpuDiffRenderTorch(gpu_system, self.device)

        self.image_channels = 4
        self.kernel = torch.tensor([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype = torch.float16, device = self.device) / 64
        self.kernel = self.kernel.expand(self.image_channels, 1, self.kernel.shape[0], self.kernel.shape[1])

    def Destroy(self):
        del self.kernel
        del self.gpu_dr
        del self.device
        torch.cuda.empty_cache()

    def Optimize(self,
                 vtx_positions, vtx_colors, num_vertices, indices, num_indices,
                 view_images, view_proj_mtxs, transform_offsets, num_views,
                 scale, rotation, translation):
        torch.cuda.empty_cache()

        vtx_positions = np.frombuffer(vtx_positions, dtype = np.float32, count = num_vertices * 3)
        vtx_positions = torch.from_numpy(vtx_positions.copy()).to(self.device)
        vtx_positions = vtx_positions.reshape(num_vertices, 3)

        vtx_colors = np.frombuffer(vtx_colors, dtype = np.float32, count = num_vertices * 3)
        vtx_colors = torch.from_numpy(vtx_colors.copy()).to(self.device)
        vtx_colors = vtx_colors.reshape(num_vertices, 3)
        vtx_colors = vtx_colors.to(torch.float16)

        indices = np.frombuffer(indices, dtype = np.int32, count = num_indices)
        indices = torch.from_numpy(indices.copy()).to(self.device)
        indices = indices.reshape(num_indices // 3, 3)

        view_proj_mtxs = np.frombuffer(view_proj_mtxs, dtype = np.float32, count = num_views * 16)
        view_proj_mtxs = torch.from_numpy(view_proj_mtxs.copy())
        view_proj_mtxs = view_proj_mtxs.reshape(num_views, 4, 4)

        transform_offsets = np.frombuffer(transform_offsets, dtype = np.int32, count = num_views * 2)
        transform_offsets = torch.from_numpy(transform_offsets.copy())
        transform_offsets = transform_offsets.reshape(num_views, 2)

        scale = np.frombuffer(scale, dtype = np.float32, count = 3)
        scale = torch.from_numpy(scale.copy())

        rotation = np.frombuffer(rotation, dtype = np.float32, count = 4)
        rotation = torch.from_numpy(rotation.copy())

        translation = np.frombuffer(translation, dtype = np.float32, count = 3)
        translation = torch.from_numpy(translation.copy())

        rois = torch.empty(num_views, 4, dtype = torch.int32, device = "cpu")
        crop_images = []
        resolutions = []
        for i in range(0, num_views):
            cropped_data = view_images[i][0]
            cropped_x = view_images[i][1]
            cropped_y = view_images[i][2]
            cropped_width = view_images[i][3]
            cropped_height = view_images[i][4]
            image_width = view_images[i][5]
            image_height = view_images[i][6]

            roi_image = np.frombuffer(cropped_data, dtype = np.uint8, count = cropped_height * cropped_width * self.image_channels)
            roi_image = torch.from_numpy(roi_image.copy())
            roi_image = roi_image.reshape(cropped_height, cropped_width, self.image_channels)

            rois[i] = torch.tensor([cropped_x, cropped_y, cropped_x + cropped_width, cropped_y + cropped_height])

            image = functional.pad(roi_image, (0, 0, rois[i][0], image_width - rois[i][2], rois[i][1], image_height - rois[i][3]), "constant", 0)
            image = image.to(self.device)
            image = image.to(torch.float16)
            image /= 255

            if self.downsampling:
                rois[i] = (rois[i] + 1) // 2
                transform_offsets[i] = (transform_offsets[i] + 1) // 2
                resolution = ((image_width + 1) // 2, (image_height + 1) // 2)
                image = self.DownsampleImage(image.unsqueeze(0)).squeeze(0)
            else:
                resolution = (image_width, image_height)

            crop_img = image[rois[i][1] : rois[i][3], rois[i][0] : rois[i][2], :]
            crop_img = crop_img.contiguous()
            crop_img = crop_img * torch.clamp(crop_img[..., -1 : ], 0, 1)
            crop_images.append(crop_img)
            resolutions.append(resolution)

        merged_roi = rois[0].clone()
        for i in range(1, num_views):
            merged_roi[0] = min(merged_roi[0], rois[i][0])
            merged_roi[1] = min(merged_roi[1], rois[i][1])
            merged_roi[2] = max(merged_roi[2], rois[i][2])
            merged_roi[3] = max(merged_roi[3], rois[i][3])

        cropped_resolution = (merged_roi[2] - merged_roi[0], merged_roi[3] - merged_roi[1])

        viewports = [None] * num_views
        for i in range(0, num_views):
            viewports[i] = Viewport()
            viewports[i].left = -merged_roi[0] + transform_offsets[i][0]
            viewports[i].top = -merged_roi[1] + transform_offsets[i][1]
            viewports[i].width = resolutions[i][0]
            viewports[i].height = resolutions[i][1]

        cropped_roi = torch.empty(num_views, 4, dtype = torch.int32, device = "cpu")
        for i in range(0, num_views):
            cropped_roi[i] = rois[i] - torch.cat([merged_roi[0 : 2], merged_roi[0 : 2]])

        vtx_colors = torch.cat([vtx_colors, torch.ones([vtx_colors.shape[0], 1], dtype = torch.float32, device = self.device)], axis = 1)

        scale, rotation, translation = self.FitTransform(scale, rotation, translation, vtx_positions, vtx_colors, indices, crop_images, view_proj_mtxs, viewports, cropped_roi, cropped_resolution)
        return (scale.cpu().numpy().tobytes(), rotation.cpu().numpy().tobytes(), translation.cpu().numpy().tobytes())

    def DownsampleImage(self, img):
        img = functional.conv2d(img.permute(0, 3, 1, 2), self.kernel, padding = 1, stride = 2, groups = img.shape[-1])
        img = img.permute(0, 2, 3, 1)
        return img

    def Render(self, model_mtx, view_proj_mtx, viewport, vtx_positions, vtx_colors, indices, opposite_vertices, resolution, roi):
        mvp_mtx = torch.matmul(model_mtx, view_proj_mtx).to(self.device)

        pos_w = torch.cat([vtx_positions, torch.ones([vtx_positions.shape[0], 1], device = self.device)], axis = 1)
        pos_clip = torch.matmul(pos_w, mvp_mtx)

        barycentric, prim_id = self.gpu_dr.Rasterize(pos_clip, indices, resolution, viewport)
        image = self.gpu_dr.Interpolate(vtx_colors, barycentric, prim_id, indices)
        image = self.gpu_dr.AntiAlias(image, prim_id, pos_clip, indices, viewport, opposite_vertices)

        image = image.to(torch.float16)
        image = image.squeeze(0)
        image = image[roi[1] : roi[3], roi[0] : roi[2], :]
        return image.contiguous()

    def FitTransform(self,
                     scale, rotation, translation,
                     vtx_positions, vtx_colors, indices,
                     crop_images, view_proj_mtxs, viewports, rois,
                     resolutions, num_iter = 2000):
        num_images = len(crop_images)
        criterion = nn.MSELoss()

        interval = 100
        lr_base = 1e-2
        lr_ramp = 1e-3

        scale_opt = nn.Parameter(scale)
        rotation_opt = nn.Parameter(rotation)
        translation_opt = nn.Parameter(translation)

        parameters = [scale_opt, rotation_opt, translation_opt]
        optimizer = optim.Adam(parameters, betas = (0.9, 0.999), lr = lr_base)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda x: max(lr_ramp, 10 ** (-x * 0.0001)))

        loss_best = 1e10
        scale_best = scale_opt.detach().clone()
        rotation_best = rotation_opt.detach().clone()
        translation_best = translation_opt.detach().clone()

        opposite_vertices = self.gpu_dr.AntiAliasConstructOppositeVertices(indices)

        loss_sum = torch.zeros(1, dtype = torch.float32, device = self.device)
        n = 0
        for it in range(num_iter + 1):
            img_idx = random.randint(0, num_images - 1)

            model_mtx_opt = ComposeMatrix(scale_opt, rotation_opt, translation_opt)
            color_opt = self.Render(model_mtx_opt, view_proj_mtxs[img_idx], viewports[img_idx], vtx_positions, vtx_colors, indices, opposite_vertices, resolutions, rois[img_idx])
            crop_img = crop_images[img_idx]

            loss = criterion(crop_img, color_opt)

            loss_val = loss.item()
            if (loss_val < loss_best) and (loss_val > 0):
                scale_best = scale_opt.detach().clone()
                rotation_best = rotation_opt.detach().clone()
                translation_best = translation_opt.detach().clone()
                loss_best = loss_val

            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                rotation_opt[:] = NormalizeQuat(rotation_opt)

            loss_sum += loss
            n += 1

            if it % interval == 0:
                print(f"Iteration {it}, loss {(loss_sum.item() / n):.7f}, loss best {loss_best:.7f}")
                loss_sum = torch.zeros(1, dtype = torch.float32, device = self.device)
                n = 0

        return scale_best, rotation_best, translation_best
