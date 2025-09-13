# Copyright (c) 2024 Minmin Gong
#

import random

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

from PythonSystem import ComputeDevice, GeneralDevice, PurgeTorchCache, TensorFromBytes, TensorToBytes
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

def LogNextPowerOf2(x):
    assert(x > 0)
    return (x - 1).bit_length()

class DiffOptimizer:
    def __init__(self, gpu_system):
        self.downsampling = True

        self.device = ComputeDevice()
        self.gpu_dr = GpuDiffRenderTorch(gpu_system, self.device)

        self.image_channels = 4
        self.kernel = torch.tensor([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype = torch.float16, device = self.device) / 64
        self.kernel = self.kernel.expand(self.image_channels, 1, self.kernel.shape[0], self.kernel.shape[1])

    def Destroy(self):
        del self.kernel
        del self.gpu_dr
        PurgeTorchCache()

    def OptimizeTransform(self,
                          vtx_positions, vtx_colors, num_vertices, indices, num_indices,
                          view_images, view_proj_mtxs, transform_offsets, num_views,
                          scale, rotation, translation):
        PurgeTorchCache()

        vtx_positions = TensorFromBytes(vtx_positions, torch.float32, num_vertices * 3, self.device)
        vtx_positions = vtx_positions.reshape(num_vertices, 3)

        vtx_colors = TensorFromBytes(vtx_colors, torch.float32, num_vertices * 3, self.device)
        vtx_colors = vtx_colors.reshape(num_vertices, 3)

        indices = TensorFromBytes(indices, torch.int32, num_indices, self.device)
        indices = indices.reshape(num_indices // 3, 3)

        view_proj_mtxs = TensorFromBytes(view_proj_mtxs, torch.float32, num_views * 4 * 4)
        view_proj_mtxs = view_proj_mtxs.reshape(num_views, 4, 4)

        transform_offsets = TensorFromBytes(transform_offsets, torch.int32, num_views * 2)
        transform_offsets = transform_offsets.reshape(num_views, 2)

        scale = TensorFromBytes(scale, torch.float32, 3)
        rotation = TensorFromBytes(rotation, torch.float32, 4)
        translation = TensorFromBytes(translation, torch.float32, 3)

        rois = torch.empty(num_views, 4, dtype = torch.int32, device = GeneralDevice())
        crop_images = []
        resolutions = []
        for i in range(0, num_views):
            roi_image = view_images[i][0].squeeze(0)
            cropped_x = view_images[i][1]
            cropped_y = view_images[i][2]
            cropped_width = roi_image.shape[1]
            cropped_height = roi_image.shape[0]
            image_width = view_images[i][3]
            image_height = view_images[i][4]

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

            rois[i][0] = max(rois[i][0], 0)
            rois[i][1] = max(rois[i][1], 0)
            rois[i][2] = min(rois[i][2], image.shape[1])
            rois[i][3] = min(rois[i][3], image.shape[0])

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

        cropped_roi = torch.empty(num_views, 4, dtype = torch.int32, device = GeneralDevice())
        for i in range(0, num_views):
            cropped_roi[i] = rois[i] - torch.cat([merged_roi[0 : 2], merged_roi[0 : 2]])

        vtx_positions = torch.cat([vtx_positions, torch.ones([vtx_positions.shape[0], 1], dtype = torch.float32, device = self.device)], axis = 1)
        vtx_colors = torch.cat([vtx_colors, torch.ones([vtx_colors.shape[0], 1], dtype = torch.float32, device = self.device)], axis = 1)

        scale, rotation, translation = self.FitTransform(scale, rotation, translation, vtx_positions, vtx_colors, indices, crop_images, view_proj_mtxs, viewports, cropped_roi, cropped_resolution)
        return (TensorToBytes(scale), TensorToBytes(rotation), TensorToBytes(translation))

    def DownsampleImage(self, img):
        img = functional.conv2d(img.permute(0, 3, 1, 2), self.kernel, padding = 1, stride = 2, groups = img.shape[-1])
        img = img.permute(0, 2, 3, 1)
        return img

    def Render(self, mvp_mtx, viewport, vtx_positions, indices, opposite_vertices, resolution, roi, **kwargs):
        pos_clip = torch.matmul(vtx_positions, mvp_mtx)

        barycentric, prim_id = self.gpu_dr.Rasterize(pos_clip, indices, resolution, viewport)
        if "vtx_colors" in kwargs:
            vtx_colors = kwargs["vtx_colors"]
            image = self.gpu_dr.Interpolate(vtx_colors, barycentric, prim_id, indices)
        else:
            vtx_uv = kwargs["vtx_uv"]
            texture = kwargs["texture"]
            uv = self.gpu_dr.Interpolate(vtx_uv, barycentric, prim_id, indices)
            image = self.gpu_dr.Texture(texture.unsqueeze(0), prim_id, uv, filter = "linear", address_mode = "clamp")
        image = self.gpu_dr.AntiAlias(image, prim_id, pos_clip, indices, viewport, opposite_vertices)

        image = image.squeeze(0)
        image = image[roi[1] : roi[3], roi[0] : roi[2], :]
        image = image.to(torch.float16)
        return image.contiguous()

    def FitTransform(self,
                     scale, rotation, translation,
                     vtx_positions, vtx_colors, indices,
                     crop_images, view_proj_mtxs, viewports, rois,
                     resolutions, num_iter = 300):
        num_images = len(crop_images)
        criterion = nn.MSELoss()

        interval = 100
        lr_base = 1e-2

        scale_opt = nn.Parameter(scale)
        rotation_opt = nn.Parameter(rotation)
        translation_opt = nn.Parameter(translation)

        parameters = [scale_opt, rotation_opt, translation_opt]
        optimizer = optim.Adam(parameters, betas = (0.9, 0.999), lr = lr_base)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5)

        loss_best = 1e10
        scale_best = scale_opt.detach().clone()
        rotation_best = rotation_opt.detach().clone()
        translation_best = translation_opt.detach().clone()

        opposite_vertices = self.gpu_dr.AntiAliasConstructOppositeVertices(indices)

        loss_sum = 0.0
        n = 0
        for it in range(num_iter + 1):
            img_idx = random.randint(0, num_images - 1)

            model_mtx_opt = ComposeMatrix(scale_opt, rotation_opt, translation_opt)
            mvp_mtx = torch.matmul(model_mtx_opt, view_proj_mtxs[img_idx]).to(self.device)
            color_opt = self.Render(mvp_mtx, viewports[img_idx], vtx_positions, indices, opposite_vertices, resolutions, rois[img_idx], vtx_colors = vtx_colors)
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
            scheduler.step(loss)

            with torch.no_grad():
                rotation_opt[:] = NormalizeQuat(rotation_opt)

            loss_sum += loss_val
            n += 1

            if it % interval == 0:
                print(f"Iteration {it}, loss {(loss_sum / n):.7f}, loss best {loss_best:.7f}")
                loss_sum = 0.0
                n = 0

        return scale_best, rotation_best, translation_best

    def OptimizeTexture(self,
                        vtx_positions, vtx_uv, num_vertices, indices, num_indices,
                        view_images, mvp_mtxs, transform_offsets, num_views,
                        texture_data, texture_width, texture_height, mask_tex_data):
        PurgeTorchCache()

        vtx_positions = TensorFromBytes(vtx_positions, torch.float32, num_vertices * 3, self.device)
        vtx_positions = vtx_positions.reshape(num_vertices, 3)

        vtx_uv = TensorFromBytes(vtx_uv, torch.float32, num_vertices * 2, self.device)
        vtx_uv = vtx_uv.reshape(num_vertices, 2)

        indices = TensorFromBytes(indices, torch.int32, num_indices, self.device)
        indices = indices.reshape(num_indices // 3, 3)

        mvp_mtxs = TensorFromBytes(mvp_mtxs, torch.float32, num_views * 4 * 4)
        mvp_mtxs = mvp_mtxs.reshape(num_views, 4, 4)
        mvp_mtxs = mvp_mtxs.to(self.device)

        transform_offsets = TensorFromBytes(transform_offsets, torch.int32, num_views * 2)
        transform_offsets = transform_offsets.reshape(num_views, 2)

        rois = torch.empty(num_views, 4, dtype = torch.int32, device = GeneralDevice())
        crop_images = []
        resolutions = []
        for i in range(0, num_views):
            roi_image = view_images[i][0].squeeze(0)
            cropped_x = view_images[i][1]
            cropped_y = view_images[i][2]
            cropped_width = roi_image.shape[1]
            cropped_height = roi_image.shape[0]
            image_width = view_images[i][3]
            image_height = view_images[i][4]

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

            rois[i][0] = max(rois[i][0], 0)
            rois[i][1] = max(rois[i][1], 0)
            rois[i][2] = min(rois[i][2], image.shape[1])
            rois[i][3] = min(rois[i][3], image.shape[0])

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

        cropped_roi = torch.empty(num_views, 4, dtype = torch.int32, device = GeneralDevice())
        for i in range(0, num_views):
            cropped_roi[i] = rois[i] - torch.cat([merged_roi[0 : 2], merged_roi[0 : 2]])

        vtx_positions = torch.cat([vtx_positions, torch.ones([vtx_positions.shape[0], 1], dtype = torch.float32, device = self.device)], axis = 1)

        texture = TensorFromBytes(texture_data, torch.uint8, texture_height * texture_width * 4, self.device)
        texture = texture.reshape(texture_height, texture_width, 4)
        texture = texture.to(torch.float32).contiguous()
        texture /= 255.0

        mask_tex = TensorFromBytes(mask_tex_data, torch.uint8, texture_height * texture_width, self.device)
        mask_tex = mask_tex.reshape(texture_height, texture_width)
        mask_tex = (mask_tex != 0)
        mask_tex = mask_tex.contiguous()

        max_mip_level = LogNextPowerOf2(max(texture_width, texture_height))

        texture = self.FitTexture(texture, mask_tex, max_mip_level, vtx_positions, vtx_uv, indices, crop_images, mvp_mtxs, viewports, cropped_roi, cropped_resolution)
        texture = (texture * 255.0).clamp(0, 255).to(torch.uint8)
        return TensorToBytes(texture)

    def Dilate(self, image, mask, times):
        image[:, :, 3] *= mask
        kernel = torch.ones(1, 1, 3, 3, dtype = torch.float32, device = image.device) / 9.0
        kernel_rgb = torch.eye(3, 3, dtype = torch.float32, device = image.device).view(3, 3, 1, 1) * kernel

        for i in range(times):
            valid_mask = image[:, :, 3] > 0.5  # Shape: [H, W]
            invalid_mask = ~valid_mask  # Shape: [H, W]

            rgb = image[:, :, 0 : 3].permute(2, 0, 1).unsqueeze(0)  # Shape: [1, 3, H, W]
            valid_mask_float = valid_mask.float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]

            rgb_valid = rgb * valid_mask_float  # Shape: [1, 3, H, W]
            rgb_sum = torch.nn.functional.conv2d(rgb_valid, kernel_rgb, padding = 1)  # Shape: [1, 3, H, W]
            valid_count = torch.nn.functional.conv2d(valid_mask_float, kernel, padding = 1)  # Shape: [1, 1, H, W]
            rgb_avg = rgb_sum / valid_count.clamp(min = 1e-6)  # Shape: [1, 3, H, W]

            rgb_avg = rgb_avg.squeeze(0).permute(1, 2, 0)  # Shape: [H, W, 3]
            valid_count = valid_count.squeeze(0).squeeze(0)  # Shape: [H, W]

            # Set alpha to 1.0 for updated pixels
            rgba_avg = torch.cat([rgb_avg, torch.ones_like(valid_count, device = image.device).unsqueeze(-1)], dim = -1)  # Shape: [H, W, 4]

            update_mask = invalid_mask & (valid_count > 0)  # Only update invalid pixels with valid neighbors
            image[update_mask] = rgba_avg[update_mask]

        return image

    def FitTexture(self,
                   texture, mask_tex, max_mip_level,
                   vtx_positions, vtx_uv, indices,
                   crop_images, mvp_mtxs, viewports, rois,
                   resolutions, num_iter = 500):
        num_images = len(crop_images)
        criterion = torch.nn.MSELoss()

        interval = 100
        lr_base = 1e-2

        texture_opt = torch.nn.Parameter(texture)

        parameters = [texture_opt]
        optimizer = torch.optim.Adam(parameters, betas = (0.9, 0.999), lr = lr_base)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.85)

        loss_best = 1e10
        texture_best = texture_opt.detach().clone()

        opposite_vertices = self.gpu_dr.AntiAliasConstructOppositeVertices(indices)

        loss_sum = 0
        n = 0
        for it in range(num_iter + 1):
            img_idx = random.randint(0, num_images - 1)

            color_opt = self.Render(mvp_mtxs[img_idx], viewports[img_idx], vtx_positions, indices, opposite_vertices, resolutions, rois[img_idx], vtx_uv = vtx_uv, texture = texture_opt)
            crop_img = crop_images[img_idx]

            loss = criterion(crop_img, color_opt)

            loss_val = loss.item()
            if (loss_val < loss_best) and (loss_val > 0):
                texture_best = texture_opt.detach().clone()
                loss_best = loss_val

            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            with torch.no_grad():
                texture_opt[:] = self.Dilate(texture_opt, mask_tex, max_mip_level)

            loss_sum += loss_val
            n += 1

            if it % interval == 0:
                print(f"Iteration {it}, loss {(loss_sum / n):.7f}, loss best {loss_best:.7f}")
                loss_sum = 0.0
                n = 0

        return texture_best
