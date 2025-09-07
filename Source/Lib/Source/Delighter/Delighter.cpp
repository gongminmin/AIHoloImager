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

        GpuTexture2D Process(GpuCommandList& cmd_list, const GpuTexture2D& image)
        {
            PerfRegion process_perf(aihi_.PerfProfilerInstance(), "Delighter process");

            auto& gpu_system = aihi_.GpuSystemInstance();

            const uint32_t width = image.Width(0);
            const uint32_t height = image.Height(0);

            auto& tensor_converter = aihi_.TensorConverterInstance();

            PyObjectPtr roi_tensor;
            {
                PythonSystem::GilGuard guard;
                roi_tensor = MakePyObjectPtr(tensor_converter.ConvertPy(cmd_list, image));
            }

            {
                PerfRegion wait_perf(aihi_.PerfProfilerInstance(), "Wait for init");
                py_init_future_.wait();
            }

            GpuTexture2D delighted_tex;
            {
                PythonSystem::GilGuard guard;

                auto& python_system = aihi_.PythonSystemInstance();
                auto args = python_system.MakeTuple(4);
                {
                    python_system.SetTupleItem(*args, 0, std::move(roi_tensor));

                    python_system.SetTupleItem(*args, 1, python_system.MakeObject(width));
                    python_system.SetTupleItem(*args, 2, python_system.MakeObject(height));
                    python_system.SetTupleItem(*args, 3, python_system.MakeObject(FormatChannels(image.Format())));
                }

                const auto output_roi_image = python_system.CallObject(*delighter_process_method_, *args);
                tensor_converter.ConvertPy(
                    cmd_list, *output_roi_image, delighted_tex, GpuFormat::RGBA8_UNorm, GpuResourceFlag::UnorderedAccess, L"delighted_tex");

                {
                    constexpr uint32_t BlockDim = 16;

                    GpuShaderResourceView image_srv(gpu_system, image);

                    GpuUnorderedAccessView delighted_uav(gpu_system, delighted_tex);

                    GpuConstantBufferOfType<MergeMaskConstantBuffer> merge_mask_cb(gpu_system, L"merge_mask_cb");
                    merge_mask_cb->dest_size.x = width;
                    merge_mask_cb->dest_size.y = height;
                    merge_mask_cb.UploadStaging();

                    const GpuConstantBuffer* cbs[] = {&merge_mask_cb};
                    const GpuShaderResourceView* srvs[] = {&image_srv};
                    GpuUnorderedAccessView* uavs[] = {&delighted_uav};
                    const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                    cmd_list.Compute(merge_mask_pipeline_, DivUp(width, BlockDim), DivUp(height, BlockDim), 1, shader_binding);
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

    GpuTexture2D Delighter::Process(GpuCommandList& cmd_list, const GpuTexture2D& image)
    {
        return impl_->Process(cmd_list, image);
    }
} // namespace AIHoloImager
