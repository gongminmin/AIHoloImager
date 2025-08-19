// Copyright (c) 2025 Minmin Gong
//

#include "Delighter.hpp"

#include <future>

#include "Util/PerfProfiler.hpp"

#include "CompiledShader/Delighter/MergeMaskCs.h"

namespace AIHoloImager
{
    class Delighter::Impl
    {
    public:
        explicit Impl(AIHoloImagerInternal& aihi) : aihi_(aihi)
        {
            PerfRegion init_perf(aihi_.PerfProfilerInstance(), "Delighter generator init");

            py_init_future_ = std::async(std::launch::async, [this] {
                PerfRegion init_async_perf(aihi_.PerfProfilerInstance(), "Delighter init (async)");

                PythonSystem::GilGuard guard;

                auto& python_system = aihi_.PythonSystemInstance();

                delighter_module_ = python_system.Import("Delighter");
                delighter_class_ = python_system.GetAttr(*delighter_module_, "Delighter");
                delighter_ = python_system.CallObject(*delighter_class_);
                delighter_process_method_ = python_system.GetAttr(*delighter_, "Process");
            });

            auto& gpu_system = aihi_.GpuSystemInstance();

            {
                const ShaderInfo shader = {MergeMaskCs_shader, 1, 1, 1};
                merge_mask_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
        }

        ~Impl()
        {
            PerfRegion destroy_perf(aihi_.PerfProfilerInstance(), "Delighter destroy");

            py_init_future_.wait();

            PythonSystem::GilGuard guard;

            auto& python_system = aihi_.PythonSystemInstance();
            auto delighter_destroy_method = python_system.GetAttr(*delighter_, "Destroy");
            python_system.CallObject(*delighter_destroy_method);

            delighter_destroy_method.reset();
            delighter_process_method_.reset();
            delighter_.reset();
            delighter_class_.reset();
            delighter_module_.reset();
        }

        GpuTexture2D Process(GpuCommandList& cmd_list, const GpuTexture2D& image, const glm::uvec4& roi, glm::uvec2& offset)
        {
            PerfRegion process_perf(aihi_.PerfProfilerInstance(), "Delighter process");

            constexpr uint32_t Gap = 32;
            glm::uvec4 expanded_roi;
            expanded_roi.x = std::max(static_cast<uint32_t>(std::floor(roi.x)) - Gap, 0U);
            expanded_roi.y = std::max(static_cast<uint32_t>(std::floor(roi.y)) - Gap, 0U);
            expanded_roi.z = std::min(static_cast<uint32_t>(std::ceil(roi.z)) + Gap, image.Width(0));
            expanded_roi.w = std::min(static_cast<uint32_t>(std::ceil(roi.w)) + Gap, image.Height(0));

            offset = glm::uvec2(expanded_roi.x, expanded_roi.y);

            auto& gpu_system = aihi_.GpuSystemInstance();

            const uint32_t roi_width = expanded_roi.z - expanded_roi.x;
            const uint32_t roi_height = expanded_roi.w - expanded_roi.y;
            const uint32_t fmt_size = FormatSize(image.Format());
            GpuTexture2D roi_image(gpu_system, roi_width, roi_height, 1, image.Format(), GpuResourceFlag::UnorderedAccess, L"roi_image");
            cmd_list.Copy(roi_image, 0, 0, 0, 0, image, 0, GpuBox{expanded_roi.x, expanded_roi.y, 0, expanded_roi.z, expanded_roi.w, 1});

            const uint32_t roi_data_size = roi_height * roi_width * fmt_size;
            auto roi_data = std::make_unique<std::byte[]>(roi_data_size);
            const auto roi_rb_future = cmd_list.ReadBackAsync(roi_image, 0, roi_data.get(), roi_data_size);

            {
                PerfRegion wait_perf(aihi_.PerfProfilerInstance(), "Wait for init");
                py_init_future_.wait();
            }
            roi_rb_future.wait();

            GpuTexture2D delighted_tex(
                gpu_system, roi_width, roi_height, 1, GpuFormat::RGBA8_UNorm, GpuResourceFlag::UnorderedAccess, L"delighted_tex");
            {
                PythonSystem::GilGuard guard;

                auto& python_system = aihi_.PythonSystemInstance();
                auto args = python_system.MakeTuple(4);
                {
                    auto py_image = python_system.MakeObject(std::span<const std::byte>(roi_data.get(), roi_data_size));
                    python_system.SetTupleItem(*args, 0, std::move(py_image));

                    python_system.SetTupleItem(*args, 1, python_system.MakeObject(roi_width));
                    python_system.SetTupleItem(*args, 2, python_system.MakeObject(roi_height));
                    python_system.SetTupleItem(*args, 3, python_system.MakeObject(FormatChannels(roi_image.Format())));
                }

                const auto output_roi_image = python_system.CallObject(*delighter_process_method_, *args);

                cmd_list.Upload(delighted_tex, 0,
                    [roi_width, roi_height, &python_system, &output_roi_image](
                        void* dst_data, uint32_t row_pitch, [[maybe_unused]] uint32_t slice_pitch) {
                        std::byte* dst = reinterpret_cast<std::byte*>(dst_data);
                        const std::byte* src = python_system.ToBytes(*output_roi_image).data();
                        for (uint32_t y = 0; y < roi_height; ++y)
                        {
                            for (uint32_t x = 0; x < roi_width; ++x)
                            {
                                const uint32_t dst_img_offset = y * row_pitch + x * 4;
                                const uint32_t src_img_offset = (y * roi_width + x) * 3;
                                dst[dst_img_offset + 0] = src[src_img_offset + 0];
                                dst[dst_img_offset + 1] = src[src_img_offset + 1];
                                dst[dst_img_offset + 2] = src[src_img_offset + 2];
                            }
                        }
                    });

                {
                    constexpr uint32_t BlockDim = 16;

                    GpuShaderResourceView cropped_srv(gpu_system, roi_image);

                    GpuUnorderedAccessView delighted_uav(gpu_system, delighted_tex);

                    auto merge_mask_cb = GpuConstantBufferOfType<MergeMaskConstantBuffer>(gpu_system, L"merge_mask_cb");
                    merge_mask_cb->dest_size.x = roi_width;
                    merge_mask_cb->dest_size.y = roi_height;
                    merge_mask_cb.UploadStaging();

                    const GpuConstantBuffer* cbs[] = {&merge_mask_cb};
                    const GpuShaderResourceView* srvs[] = {&cropped_srv};
                    GpuUnorderedAccessView* uavs[] = {&delighted_uav};
                    const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                    cmd_list.Compute(merge_mask_pipeline_, DivUp(roi_width, BlockDim), DivUp(roi_height, BlockDim), 1, shader_binding);
                }
            }

            return delighted_tex;
        }

    private:
        AIHoloImagerInternal& aihi_;

        PyObjectPtr delighter_module_;
        PyObjectPtr delighter_class_;
        PyObjectPtr delighter_;
        PyObjectPtr delighter_process_method_;
        std::future<void> py_init_future_;

        struct MergeMaskConstantBuffer
        {
            glm::uvec2 dest_size;
            uint32_t padding[2];
        };
        GpuComputePipeline merge_mask_pipeline_;
    };

    Delighter::Delighter(AIHoloImagerInternal& aihi) : impl_(std::make_unique<Impl>(aihi))
    {
    }

    Delighter::~Delighter() noexcept = default;

    Delighter::Delighter(Delighter&& other) noexcept = default;
    Delighter& Delighter::operator=(Delighter&& other) noexcept = default;

    GpuTexture2D Delighter::Process(GpuCommandList& cmd_list, const GpuTexture2D& image, const glm::uvec4& roi, glm::uvec2& offset)
    {
        return impl_->Process(cmd_list, image, roi, offset);
    }
} // namespace AIHoloImager
