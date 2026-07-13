// Copyright (c) 2025-2026 Minmin Gong
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
                const ShaderInfo shader = {DEFINE_SHADER(MergeMaskCs)};
                merge_mask_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
        }

        ~Impl()
        {
            PerfRegion destroy_perf(aihi_.PerfProfilerInstance(), "Delighter destroy");

            if (!py_init_finished_)
            {
                py_init_future_.wait();
            }

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

        void Process(AIHoloImagerInternal::ProjectionDesc& projection)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();
            auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Compute);

            PerfRegion process_perf(aihi_.PerfProfilerInstance(), "Delighter process", &cmd_list);

            const uint32_t width = projection.image->Width(0);
            const uint32_t height = projection.image->Height(0);

            auto& tensor_converter = aihi_.TensorConverterInstance();

            PyObjectPtr roi_tensor;
            {
                PythonSystem::GilGuard guard;
                roi_tensor = MakePyObjectPtr(tensor_converter.ConvertPy(cmd_list, *projection.image));
            }

            if (!py_init_finished_)
            {
                PerfRegion wait_perf(aihi_.PerfProfilerInstance(), "Wait for init", &cmd_list);
                py_init_future_.wait();

                py_init_finished_ = true;
            }

            GpuTexture2D delighted_tex;
            {
                PythonSystem::GilGuard guard;

                auto& python_system = aihi_.PythonSystemInstance();

                const auto output_roi_image = python_system.CallObject(*delighter_process_method_, std::move(roi_tensor));
                tensor_converter.ConvertPy(cmd_list, *output_roi_image, delighted_tex, GpuFormat::RGBA8_UNorm_SRGB,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess, "delighted_tex");

                {
                    constexpr uint32_t BlockDim = 16;

                    const GpuShaderResourceView image_srv(gpu_system, *projection.image);

                    GpuUnorderedAccessView delighted_uav(gpu_system, delighted_tex, ToLinearFormat(delighted_tex.Format()));

                    GpuConstantBufferOfType<MergeMaskConstantBuffer> merge_mask_cb(gpu_system, "merge_mask_cb");
                    merge_mask_cb->dest_size = {width, height};
                    merge_mask_cb.UploadStaging();

                    const GpuConstantBufferView merge_mask_cbv(gpu_system, merge_mask_cb);

                    std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                        {"param_cb", &merge_mask_cbv},
                    };
                    std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                        {"input_tex", &image_srv},
                    };
                    std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                        {"delighted_tex", &delighted_uav},
                    };
                    const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                    cmd_list.Compute(merge_mask_pipeline_, {DivUp(width, BlockDim), DivUp(height, BlockDim), 1}, shader_binding);
                }
            }

            process_perf.End();
            gpu_system.Execute(std::move(cmd_list));

            *projection.image = std::move(delighted_tex);
        }

    private:
        AIHoloImagerInternal& aihi_;

        PyObjectPtr delighter_module_;
        PyObjectPtr delighter_class_;
        PyObjectPtr delighter_;
        PyObjectPtr delighter_process_method_;
        std::future<void> py_init_future_;
        bool py_init_finished_ = false;

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

    void Delighter::Process(AIHoloImagerInternal::ProjectionDesc& projection)
    {
        impl_->Process(projection);
    }
} // namespace AIHoloImager
