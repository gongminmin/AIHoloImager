// Copyright (c) 2024-2025 Minmin Gong
//

#include "MaskGenerator.hpp"

#include <cstddef>
#include <future>
#include <span>

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuShader.hpp"
#include "Gpu/GpuTexture.hpp"
#include "Util/PerfProfiler.hpp"

#include "CompiledShader/MaskGen/CalcBBoxCs.h"
#include "CompiledShader/MaskGen/DownsampleCs.h"
#include "CompiledShader/MaskGen/ErosionDilationCs.h"
#include "CompiledShader/MaskGen/GaussianBlurCs.h"
#include "CompiledShader/MaskGen/MergeMaskCs.h"
#include "CompiledShader/MaskGen/NormalizeImageCs.h"
#include "CompiledShader/MaskGen/StatImageCs.h"
#include "CompiledShader/MaskGen/StatPredCs.h"
#include "CompiledShader/MaskGen/UpsampleCs.h"

namespace AIHoloImager
{
    class MaskGenerator::Impl
    {
    public:
        explicit Impl(AIHoloImagerInternal& aihi) : aihi_(aihi)
        {
            PerfRegion init_perf(aihi_.PerfProfilerInstance(), "Mask generator init");

            py_init_future_ = std::async(std::launch::async, [this] {
                PerfRegion init_async_perf(aihi_.PerfProfilerInstance(), "Mask generator init (async)");

                PythonSystem::GilGuard guard;

                auto& python_system = aihi_.PythonSystemInstance();

                mask_generator_module_ = python_system.Import("MaskGenerator");
                mask_generator_class_ = python_system.GetAttr(*mask_generator_module_, "MaskGenerator");
                mask_generator_ = python_system.CallObject(*mask_generator_class_);
                mask_generator_gen_method_ = python_system.GetAttr(*mask_generator_, "Gen");
            });

            auto& gpu_system = aihi_.GpuSystemInstance();

            {
                const ShaderInfo shader = {DownsampleCs_shader, 1, 1, 1};
                downsample_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {StatImageCs_shader, 1, 1, 1};
                stat_image_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {NormalizeImageCs_shader, 1, 2, 1};
                normalize_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {CalcBBoxCs_shader, 1, 1, 1};
                calc_bbox_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {StatPredCs_shader, 1, 1, 1};
                stat_pred_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {UpsampleCs_shader, 1, 2, 1};
                upsample_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {ErosionDilationCs_shader, 1, 1, 1};
                erosion_dilation_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {GaussianBlurCs_shader, 1, 1, 1};
                gaussian_blur_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {MergeMaskCs_shader, 1, 1, 1};
                merge_mask_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
        }

        ~Impl()
        {
            PerfRegion destroy_perf(aihi_.PerfProfilerInstance(), "Mask generator destroy");

            py_init_future_.wait();

            PythonSystem::GilGuard guard;

            auto& python_system = aihi_.PythonSystemInstance();
            auto mask_generator_destroy_method = python_system.GetAttr(*mask_generator_, "Destroy");
            python_system.CallObject(*mask_generator_destroy_method);

            mask_generator_destroy_method.reset();
            mask_generator_gen_method_.reset();
            mask_generator_.reset();
            mask_generator_class_.reset();
            mask_generator_module_.reset();
        }

        void Generate(GpuCommandList& cmd_list, GpuTexture2D& image_gpu_tex, glm::uvec4& roi)
        {
            PerfRegion generate_perf(aihi_.PerfProfilerInstance(), "Mask generator generate");

            auto& gpu_system = aihi_.GpuSystemInstance();

            const uint32_t width = image_gpu_tex.Width(0);
            const uint32_t height = image_gpu_tex.Height(0);

            const bool crop = (width > U2NetInputDim) || (height > U2NetInputDim);

            if (!mask_gpu_tex_ || (mask_gpu_tex_.Width(0) != width) || (mask_gpu_tex_.Height(0) != height))
            {
                downsampled_x_gpu_tex_ = GpuTexture2D(
                    gpu_system, U2NetInputDim, height, 1, ColorFmt, GpuResourceFlag::UnorderedAccess, L"downsampled_x_gpu_tex_");
                downsampled_gpu_tex_ = GpuTexture2D(
                    gpu_system, U2NetInputDim, U2NetInputDim, 1, ColorFmt, GpuResourceFlag::UnorderedAccess, L"downsampled_gpu_tex_");
                image_max_gpu_tex_ =
                    GpuTexture2D(gpu_system, 1, 1, 1, GpuFormat::R32_Uint, GpuResourceFlag::UnorderedAccess, L"image_max_gpu_tex_");
                normalized_gpu_tex_ = GpuTexture2D(gpu_system, U2NetInputDim, U2NetInputDim * U2NetInputChannels, 1, GpuFormat::R32_Float,
                    GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable, L"normalized_gpu_tex_");

                pred_gpu_tex_ = GpuTexture2D(gpu_system, U2NetInputDim, U2NetInputDim, 1, GpuFormat::R32_Float,
                    GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable, L"pred_gpu_tex_");
                pred_min_max_gpu_tex_ =
                    GpuTexture2D(gpu_system, 2, 1, 1, GpuFormat::R32_Uint, GpuResourceFlag::UnorderedAccess, L"pred_min_max_gpu_tex_");
                mask_gpu_tex_ = GpuTexture2D(gpu_system, width, height, 1, MaskFmt, GpuResourceFlag::UnorderedAccess, L"mask_gpu_tex_");
                mask_pingpong_gpu_tex_ =
                    GpuTexture2D(gpu_system, width, height, 1, MaskFmt, GpuResourceFlag::UnorderedAccess, L"mask_pingpong_gpu_tex_");

                if (crop)
                {
                    bbox_gpu_tex_ =
                        GpuTexture2D(gpu_system, 4, 1, 1, GpuFormat::R32_Uint, GpuResourceFlag::UnorderedAccess, L"bbox_gpu_tex_");
                }
            }

            roi = glm::uvec4(0, 0, width, height);
            this->GenMask(cmd_list, image_gpu_tex, roi, !crop, !crop);
            if (crop)
            {
                GpuConstantBufferOfType<StatPredConstantBuffer> calc_bbox_cb(gpu_system, L"calc_bbox_cb");
                calc_bbox_cb->texture_size.x = width;
                calc_bbox_cb->texture_size.y = height;
                calc_bbox_cb.UploadStaging();

                GpuUnorderedAccessView bbox_uav(gpu_system, bbox_gpu_tex_);

                {
                    const uint32_t bb_init[] = {width, height, 0, 0};
                    cmd_list.Upload(bbox_gpu_tex_, 0, bb_init, sizeof(bb_init));

                    constexpr uint32_t BlockDim = 16;

                    GpuShaderResourceView input_srv(gpu_system, image_gpu_tex);

                    const GpuConstantBuffer* cbs[] = {&calc_bbox_cb};
                    const GpuShaderResourceView* srvs[] = {&input_srv};
                    GpuUnorderedAccessView* uavs[] = {&bbox_uav};
                    const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                    cmd_list.Compute(calc_bbox_pipeline_, DivUp(width, BlockDim), DivUp(height, BlockDim), 1, shader_binding);
                }

                // TODO: Use indirect dispatch to avoid the read back
                const auto rb_future = cmd_list.ReadBackAsync(bbox_gpu_tex_, 0, &roi, sizeof(roi));
                rb_future.wait();

                const glm::uvec2 bb_min(roi.x, roi.y);
                const glm::uvec2 bb_max(roi.z, roi.w);
                const uint32_t crop_extent = std::max(std::max(bb_max.x - bb_min.x, bb_max.y - bb_min.y) + 64, U2NetInputDim) / 2;
                const glm::uvec2 crop_center = (bb_min + bb_max) / 2U;
                const glm::uvec4 square_roi = glm::clamp(glm::ivec4(crop_center - crop_extent, crop_center + crop_extent + 1U),
                    glm::ivec4(0, 0, 0, 0), glm::ivec4(width, height, width, height));

                this->GenMask(cmd_list, image_gpu_tex, square_roi, true, true);
            }
        }

    private:
        void GenMask(GpuCommandList& cmd_list, GpuTexture2D& image_gpu_tex, const glm::uvec4& roi, bool blur, bool large_model)
        {
            PerfRegion generate_perf(aihi_.PerfProfilerInstance(), std::format("U2Net ({})", large_model ? "large" : "small"));

            auto& gpu_system = aihi_.GpuSystemInstance();
            auto& python_system = aihi_.PythonSystemInstance();

            constexpr uint32_t BlockDim = 16;

            const uint32_t roi_width = roi.z - roi.x;
            const uint32_t roi_height = roi.w - roi.y;

            constexpr uint32_t InitMinMax[2] = {~0U, 0U};
            cmd_list.Upload(pred_min_max_gpu_tex_, 0, InitMinMax, sizeof(InitMinMax));

            {
                GpuShaderResourceView input_srv(gpu_system, image_gpu_tex);
                GpuUnorderedAccessView output_uav(gpu_system, downsampled_x_gpu_tex_);

                GpuConstantBufferOfType<ResizeConstantBuffer> downsample_x_cb(gpu_system, L"downsample_x_cb");
                downsample_x_cb->src_roi = roi;
                downsample_x_cb->dest_size.x = U2NetInputDim;
                downsample_x_cb->dest_size.y = roi_height;
                downsample_x_cb->scale = static_cast<float>(roi_width) / U2NetInputDim;
                downsample_x_cb->x_dir = true;
                downsample_x_cb.UploadStaging();

                const GpuConstantBuffer* cbs[] = {&downsample_x_cb};
                const GpuShaderResourceView* srvs[] = {&input_srv};
                GpuUnorderedAccessView* uavs[] = {&output_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(downsample_pipeline_, DivUp(U2NetInputDim, BlockDim), DivUp(roi_height, BlockDim), 1, shader_binding);
            }
            {
                GpuShaderResourceView input_srv(gpu_system, downsampled_x_gpu_tex_);
                GpuUnorderedAccessView output_uav(gpu_system, downsampled_gpu_tex_);

                GpuConstantBufferOfType<ResizeConstantBuffer> downsample_y_cb(gpu_system, L"downsample_y_cb");
                downsample_y_cb->src_roi.x = 0;
                downsample_y_cb->src_roi.y = 0;
                downsample_y_cb->src_roi.z = U2NetInputDim;
                downsample_y_cb->src_roi.w = roi_height;
                downsample_y_cb->dest_size.x = U2NetInputDim;
                downsample_y_cb->dest_size.y = U2NetInputDim;
                downsample_y_cb->scale = static_cast<float>(roi_height) / U2NetInputDim;
                downsample_y_cb->x_dir = false;
                downsample_y_cb.UploadStaging();

                const GpuConstantBuffer* cbs[] = {&downsample_y_cb};
                const GpuShaderResourceView* srvs[] = {&input_srv};
                GpuUnorderedAccessView* uavs[] = {&output_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(downsample_pipeline_, DivUp(U2NetInputDim, BlockDim), DivUp(U2NetInputDim, BlockDim), 1, shader_binding);
            }

            {
                GpuShaderResourceView input_srv(gpu_system, downsampled_gpu_tex_);
                GpuUnorderedAccessView max_uav(gpu_system, image_max_gpu_tex_);

                GpuConstantBufferOfType<StatPredConstantBuffer> stat_image_cb(gpu_system, L"stat_image_cb");
                stat_image_cb->texture_size.x = U2NetInputDim;
                stat_image_cb->texture_size.y = U2NetInputDim;
                stat_image_cb.UploadStaging();

                const GpuConstantBuffer* cbs[] = {&stat_image_cb};
                const GpuShaderResourceView* srvs[] = {&input_srv};
                GpuUnorderedAccessView* uavs[] = {&max_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(stat_image_pipeline_, DivUp(U2NetInputDim, BlockDim), DivUp(U2NetInputDim, BlockDim), 1, shader_binding);
            }
            {
                GpuShaderResourceView input_srv(gpu_system, downsampled_gpu_tex_);
                GpuShaderResourceView max_srv(gpu_system, image_max_gpu_tex_);
                GpuUnorderedAccessView normalized_uav(gpu_system, normalized_gpu_tex_);

                GpuConstantBufferOfType<StatPredConstantBuffer> normalize_image_cb(gpu_system, L"normalize_image_cb");
                normalize_image_cb->texture_size.x = U2NetInputDim;
                normalize_image_cb->texture_size.y = U2NetInputDim;
                normalize_image_cb.UploadStaging();

                const GpuConstantBuffer* cbs[] = {&normalize_image_cb};
                const GpuShaderResourceView* srvs[] = {&input_srv, &max_srv};
                GpuUnorderedAccessView* uavs[] = {&normalized_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(normalize_pipeline_, DivUp(U2NetInputDim, BlockDim), DivUp(U2NetInputDim, BlockDim), 1, shader_binding);
            }

            auto& tensor_converter = aihi_.TensorConverterInstance();

            PyObjectPtr normalized_image_tensor;
            {
                PythonSystem::GilGuard guard;
                normalized_image_tensor = MakePyObjectPtr(tensor_converter.ConvertPy(cmd_list, normalized_gpu_tex_));
            }

            {
                PerfRegion wait_perf(aihi_.PerfProfilerInstance(), "Wait for init");
                py_init_future_.wait();
            }

            {
                PythonSystem::GilGuard guard;

                auto args = python_system.MakeTuple(5);
                {
                    python_system.SetTupleItem(*args, 0, std::move(normalized_image_tensor));

                    python_system.SetTupleItem(*args, 1, python_system.MakeObject(U2NetInputDim));
                    python_system.SetTupleItem(*args, 2, python_system.MakeObject(U2NetInputDim));
                    python_system.SetTupleItem(*args, 3, python_system.MakeObject(U2NetInputChannels));
                    python_system.SetTupleItem(*args, 4, python_system.MakeObject(large_model));
                }

                auto py_pred = python_system.CallObject(*mask_generator_gen_method_, *args);
                tensor_converter.ConvertPy(
                    cmd_list, *py_pred, pred_gpu_tex_, GpuFormat::R32_Float, GpuResourceFlag::UnorderedAccess, L"pred_gpu_tex_");
            }

            {
                GpuShaderResourceView input_srv(gpu_system, pred_gpu_tex_);
                GpuUnorderedAccessView min_max_uav(gpu_system, pred_min_max_gpu_tex_);

                GpuConstantBufferOfType<StatPredConstantBuffer> stat_pred_cb(gpu_system, L"stat_pred_cb");
                stat_pred_cb->texture_size.x = U2NetInputDim;
                stat_pred_cb->texture_size.y = U2NetInputDim;
                stat_pred_cb.UploadStaging();

                const GpuConstantBuffer* cbs[] = {&stat_pred_cb};
                const GpuShaderResourceView* srvs[] = {&input_srv};
                GpuUnorderedAccessView* uavs[] = {&min_max_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(stat_pred_pipeline_, DivUp(U2NetInputDim, BlockDim), DivUp(U2NetInputDim, BlockDim), 1, shader_binding);
            }

            {
                GpuShaderResourceView input_srv(gpu_system, pred_gpu_tex_);
                GpuShaderResourceView min_max_srv(gpu_system, pred_min_max_gpu_tex_);
                GpuUnorderedAccessView output_uav(gpu_system, mask_pingpong_gpu_tex_);

                GpuConstantBufferOfType<ResizeConstantBuffer> upsample_x_cb(gpu_system, L"upsample_x_cb");
                upsample_x_cb->src_roi.x = 0;
                upsample_x_cb->src_roi.y = 0;
                upsample_x_cb->src_roi.z = U2NetInputDim;
                upsample_x_cb->src_roi.w = U2NetInputDim;
                upsample_x_cb->dest_size.x = roi_width;
                upsample_x_cb->dest_size.y = U2NetInputDim;
                upsample_x_cb->scale = static_cast<float>(U2NetInputDim) / roi_width;
                upsample_x_cb->x_dir = true;
                upsample_x_cb.UploadStaging();

                const GpuConstantBuffer* cbs[] = {&upsample_x_cb};
                const GpuShaderResourceView* srvs[] = {&input_srv, &min_max_srv};
                GpuUnorderedAccessView* uavs[] = {&output_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(upsample_pipeline_, DivUp(roi_width, BlockDim), DivUp(U2NetInputDim, BlockDim), 1, shader_binding);
            }
            {
                GpuShaderResourceView input_srv(gpu_system, mask_pingpong_gpu_tex_);
                GpuUnorderedAccessView output_uav(gpu_system, mask_gpu_tex_);

                GpuConstantBufferOfType<ResizeConstantBuffer> upsample_y_cb(gpu_system, L"upsample_y_cb");
                upsample_y_cb->src_roi.x = 0;
                upsample_y_cb->src_roi.y = 0;
                upsample_y_cb->src_roi.z = roi_width;
                upsample_y_cb->src_roi.w = U2NetInputDim;
                upsample_y_cb->dest_size.x = roi_width;
                upsample_y_cb->dest_size.y = roi_height;
                upsample_y_cb->scale = static_cast<float>(U2NetInputDim) / roi_height;
                upsample_y_cb->x_dir = false;
                upsample_y_cb.UploadStaging();

                const GpuConstantBuffer* cbs[] = {&upsample_y_cb};
                const GpuShaderResourceView* srvs[] = {&input_srv};
                GpuUnorderedAccessView* uavs[] = {&output_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(upsample_pipeline_, DivUp(roi_width, BlockDim), DivUp(roi_height, BlockDim), 1, shader_binding);
            }

            {
                constexpr uint32_t Kernel[][3] = {
                    {0, 1, 0},
                    {1, 1, 1},
                    {0, 1, 0},
                };

                {
                    GpuShaderResourceView input_srv(gpu_system, mask_gpu_tex_);
                    GpuUnorderedAccessView output_uav(gpu_system, mask_pingpong_gpu_tex_);

                    GpuConstantBufferOfType<ErosionDilateConstantBuffer> erosion_cb(gpu_system, L"erosion_cb");
                    erosion_cb->texture_size.x = roi_width;
                    erosion_cb->texture_size.y = roi_height;
                    erosion_cb->erosion = true;
                    erosion_cb->weights[0].x = Kernel[0][0];
                    erosion_cb->weights[1].x = Kernel[0][1];
                    erosion_cb->weights[2].x = Kernel[0][2];
                    erosion_cb->weights[3].x = Kernel[1][0];
                    erosion_cb->weights[4].x = Kernel[1][1];
                    erosion_cb->weights[5].x = Kernel[1][2];
                    erosion_cb->weights[6].x = Kernel[2][0];
                    erosion_cb->weights[7].x = Kernel[2][1];
                    erosion_cb->weights[8].x = Kernel[2][2];
                    erosion_cb.UploadStaging();

                    const GpuConstantBuffer* cbs[] = {&erosion_cb};
                    const GpuShaderResourceView* srvs[] = {&input_srv};
                    GpuUnorderedAccessView* uavs[] = {&output_uav};
                    const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                    cmd_list.Compute(
                        erosion_dilation_pipeline_, DivUp(roi_width, BlockDim), DivUp(roi_height, BlockDim), 1, shader_binding);
                }
                {
                    GpuShaderResourceView input_srv(gpu_system, mask_pingpong_gpu_tex_);
                    GpuUnorderedAccessView output_uav(gpu_system, mask_gpu_tex_);

                    GpuConstantBufferOfType<ErosionDilateConstantBuffer> dilation_cb(gpu_system, L"dilation_cb");
                    dilation_cb->texture_size.x = roi_width;
                    dilation_cb->texture_size.y = roi_height;
                    dilation_cb->erosion = false;
                    dilation_cb->weights[0].x = Kernel[0][0];
                    dilation_cb->weights[1].x = Kernel[0][1];
                    dilation_cb->weights[2].x = Kernel[0][2];
                    dilation_cb->weights[3].x = Kernel[1][0];
                    dilation_cb->weights[4].x = Kernel[1][1];
                    dilation_cb->weights[5].x = Kernel[1][2];
                    dilation_cb->weights[6].x = Kernel[2][0];
                    dilation_cb->weights[7].x = Kernel[2][1];
                    dilation_cb->weights[8].x = Kernel[2][2];
                    dilation_cb.UploadStaging();

                    const GpuConstantBuffer* cbs[] = {&dilation_cb};
                    const GpuShaderResourceView* srvs[] = {&input_srv};
                    GpuUnorderedAccessView* uavs[] = {&output_uav};
                    const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                    cmd_list.Compute(
                        erosion_dilation_pipeline_, DivUp(roi_width, BlockDim), DivUp(roi_height, BlockDim), 1, shader_binding);
                }
            }

            if (blur)
            {
                // Deviation = 2
                constexpr float Weights[] = {0.054488689f, 0.24420136f, 0.40261996f, 0.24420136f, 0.054488689f};
                static_assert(std::size(Weights) == BlurKernelRadius * 2 + 1);

                {
                    GpuShaderResourceView input_srv(gpu_system, mask_gpu_tex_);
                    GpuUnorderedAccessView output_uav(gpu_system, mask_pingpong_gpu_tex_);

                    GpuConstantBufferOfType<GaussianBlurConstantBuffer> gaussian_blur_x_cb(gpu_system, L"gaussian_blur_x_cb");
                    gaussian_blur_x_cb->texture_size.x = roi_width;
                    gaussian_blur_x_cb->texture_size.y = roi_height;
                    gaussian_blur_x_cb->x_dir = true;
                    for (uint32_t i = 0; i < BlurKernelRadius * 2 + 1; ++i)
                    {
                        gaussian_blur_x_cb->weights[i].x = Weights[i];
                    }
                    gaussian_blur_x_cb.UploadStaging();

                    const GpuConstantBuffer* cbs[] = {&gaussian_blur_x_cb};
                    const GpuShaderResourceView* srvs[] = {&input_srv};
                    GpuUnorderedAccessView* uavs[] = {&output_uav};
                    const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                    cmd_list.Compute(gaussian_blur_pipeline_, DivUp(roi_width, BlockDim), DivUp(roi_height, BlockDim), 1, shader_binding);
                }
                {
                    GpuShaderResourceView input_srv(gpu_system, mask_pingpong_gpu_tex_);
                    GpuUnorderedAccessView output_uav(gpu_system, mask_gpu_tex_);

                    GpuConstantBufferOfType<GaussianBlurConstantBuffer> gaussian_blur_y_cb(gpu_system, L"gaussian_blur_y_cb");
                    gaussian_blur_y_cb->texture_size.x = roi_width;
                    gaussian_blur_y_cb->texture_size.y = roi_height;
                    gaussian_blur_y_cb->x_dir = false;
                    for (uint32_t i = 0; i < BlurKernelRadius * 2 + 1; ++i)
                    {
                        gaussian_blur_y_cb->weights[i].x = Weights[i];
                    }
                    gaussian_blur_y_cb.UploadStaging();

                    const GpuConstantBuffer* cbs[] = {&gaussian_blur_y_cb};
                    const GpuShaderResourceView* srvs[] = {&input_srv};
                    GpuUnorderedAccessView* uavs[] = {&output_uav};
                    const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                    cmd_list.Compute(gaussian_blur_pipeline_, DivUp(roi_width, BlockDim), DivUp(roi_height, BlockDim), 1, shader_binding);
                }
            }

            {
                const uint32_t width = image_gpu_tex.Width(0);
                const uint32_t height = image_gpu_tex.Height(0);

                GpuShaderResourceView input_srv(gpu_system, mask_gpu_tex_);
                GpuUnorderedAccessView output_uav(gpu_system, image_gpu_tex);

                GpuConstantBufferOfType<MergeMaskConstantBuffer> merge_mask_cb(gpu_system, L"merge_mask_cb");
                merge_mask_cb->texture_size.x = width;
                merge_mask_cb->texture_size.y = height;
                merge_mask_cb->roi = roi;
                merge_mask_cb.UploadStaging();

                const GpuConstantBuffer* cbs[] = {&merge_mask_cb};
                const GpuShaderResourceView* srvs[] = {&input_srv};
                GpuUnorderedAccessView* uavs[] = {&output_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(merge_mask_pipeline_, DivUp(width, BlockDim), DivUp(height, BlockDim), 1, shader_binding);
            }
        }

    private:
        AIHoloImagerInternal& aihi_;

        PyObjectPtr mask_generator_module_;
        PyObjectPtr mask_generator_class_;
        PyObjectPtr mask_generator_;
        PyObjectPtr mask_generator_gen_method_;
        std::future<void> py_init_future_;

        GpuTexture2D downsampled_x_gpu_tex_;
        GpuTexture2D downsampled_gpu_tex_;
        GpuTexture2D image_max_gpu_tex_;
        GpuTexture2D normalized_gpu_tex_;
        GpuTexture2D bbox_gpu_tex_;
        GpuTexture2D pred_gpu_tex_;
        GpuTexture2D pred_min_max_gpu_tex_;
        GpuTexture2D mask_gpu_tex_;
        GpuTexture2D mask_pingpong_gpu_tex_;

        struct StatPredConstantBuffer
        {
            glm::uvec2 texture_size;
            uint32_t padding[2];
        };
        GpuComputePipeline normalize_pipeline_;
        GpuComputePipeline stat_image_pipeline_;
        GpuComputePipeline stat_pred_pipeline_;
        GpuComputePipeline calc_bbox_pipeline_;

        struct ResizeConstantBuffer
        {
            glm::uvec4 src_roi;
            glm::uvec2 dest_size;
            float scale;
            uint32_t x_dir;
        };
        GpuComputePipeline downsample_pipeline_;
        GpuComputePipeline upsample_pipeline_;

        static constexpr uint32_t ErosionDilateKernelRadius = 1;
        struct ErosionDilateConstantBuffer
        {
            glm::uvec2 texture_size;
            uint32_t erosion;
            uint32_t padding;
            glm::uvec4 weights[(ErosionDilateKernelRadius * 2 + 1) * (ErosionDilateKernelRadius * 2 + 1)];
        };
        GpuComputePipeline erosion_dilation_pipeline_;

        static constexpr uint32_t BlurKernelRadius = 2;
        struct GaussianBlurConstantBuffer
        {
            glm::uvec2 texture_size;
            uint32_t x_dir;
            uint32_t padding;
            glm::vec4 weights[BlurKernelRadius * 2 + 1];
        };
        GpuComputePipeline gaussian_blur_pipeline_;

        struct MergeMaskConstantBuffer
        {
            glm::uvec2 texture_size;
            uint32_t padding[2];
            glm::uvec4 roi;
        };
        GpuComputePipeline merge_mask_pipeline_;

        static constexpr GpuFormat MaskFmt = GpuFormat::R8_UNorm;
        static constexpr GpuFormat ColorFmt = GpuFormat::RGBA8_UNorm;
        static constexpr uint32_t U2NetInputDim = 320;
        static constexpr uint32_t U2NetInputChannels = 3;
    };

    MaskGenerator::MaskGenerator(AIHoloImagerInternal& aihi) : impl_(std::make_unique<Impl>(aihi))
    {
    }

    MaskGenerator::~MaskGenerator() noexcept = default;

    MaskGenerator::MaskGenerator(MaskGenerator&& other) noexcept = default;
    MaskGenerator& MaskGenerator::operator=(MaskGenerator&& other) noexcept = default;

    void MaskGenerator::Generate(GpuCommandList& cmd_list, GpuTexture2D& image, glm::uvec4& roi)
    {
        impl_->Generate(cmd_list, image, roi);
    }
} // namespace AIHoloImager
