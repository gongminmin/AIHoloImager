// Copyright (c) 2026 Minmin Gong
//

#include "SuperResolution.hpp"

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "CompiledShader/SuperRes/BlendCs.h"
#include "CompiledShader/SuperRes/ResizeCs.h"

namespace AIHoloImager
{
    class SuperResolution::Impl
    {
    public:
        explicit Impl(AIHoloImagerInternal& aihi) : aihi_(aihi)
        {
            py_init_future_ = std::async(std::launch::async, [this] {
                PerfRegion init_async_perf(aihi_.PerfProfilerInstance(), "SuperResolution init (async)");

                PythonSystem::GilGuard guard;

                auto& python_system = aihi_.PythonSystemInstance();

                super_res_module_ = python_system.Import("SuperResolution");
                super_res_class_ = python_system.GetAttr(*super_res_module_, "SuperResolution");
                super_res_ = python_system.CallObject(*super_res_class_);
                super_res_process_method_ = python_system.GetAttr(*super_res_, "Process");
            });

            auto& gpu_system = aihi_.GpuSystemInstance();

            {
                const ShaderInfo shader = {DEFINE_SHADER(BlendCs)};
                blend_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(ResizeCs)};
                resize_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
        }

        ~Impl()
        {
            PerfRegion destroy_perf(aihi_.PerfProfilerInstance(), "SuperResolution destroy");

            if (!py_init_finished_)
            {
                py_init_future_.wait();
            }

            PythonSystem::GilGuard guard;

            auto& python_system = aihi_.PythonSystemInstance();
            auto image_upsampler_destroy_method = python_system.GetAttr(*super_res_, "Destroy");
            python_system.CallObject(*image_upsampler_destroy_method);

            image_upsampler_destroy_method.reset();
            super_res_process_method_.reset();
            super_res_.reset();
            super_res_class_.reset();
            super_res_module_.reset();
        }

        AIHoloImagerInternal::ProjectionDesc Process(const AIHoloImagerInternal::ProjectionDesc& input, float scale)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Compute);

            const auto& image = *input.image;

            GpuTexture2D upsampled_tex = this->NnScale(cmd_list, image);
            if (std::abs(scale - upsampled_tex.Width(0) / image.Width(0)) > 1e-6f)
            {
                const uint32_t target_width = static_cast<uint32_t>(std::round(image.Width(0) * scale));
                const uint32_t target_height = static_cast<uint32_t>(std::round(image.Height(0) * scale));
                upsampled_tex = this->Resize(cmd_list, upsampled_tex, target_width, target_height);
            }

#ifdef AIHI_KEEP_INTERMEDIATES
            upsampled_tex.Transition(cmd_list, GpuResourceState::Common);
#endif

            gpu_system.Execute(std::move(cmd_list));

            AIHoloImagerInternal::ProjectionDesc ret;
            ret.image = std::make_shared<GpuTexture2D>(std::move(upsampled_tex));
            ret.view_mtx = input.view_mtx;
            ret.proj_mtx = input.proj_mtx;
            ret.full_width = static_cast<uint32_t>(std::round(input.full_width * scale));
            ret.full_height = static_cast<uint32_t>(std::round(input.full_height * scale));
            ret.vp_offset = input.vp_offset * scale;
            ret.image_offset = glm::uvec2(static_cast<uint32_t>(std::round(input.image_offset.x * scale)),
                static_cast<uint32_t>(std::round(input.image_offset.y * scale)));

            return ret;
        }

    private:
        GpuTexture2D NnScale(GpuCommandList& cmd_list, const GpuTexture2D& image)
        {
            auto& tensor_converter = aihi_.TensorConverterInstance();
            auto& gpu_system = aihi_.GpuSystemInstance();

            if (!py_init_finished_)
            {
                PerfRegion wait_perf(aihi_.PerfProfilerInstance(), "Wait for init");
                py_init_future_.wait();

                py_init_finished_ = true;
            }

            GpuTexture3D upsampled_residual_tex;
            {
                PerfRegion perf(aihi_.PerfProfilerInstance(), "Nn scale");

                PythonSystem::GilGuard guard;

                auto& python_system = aihi_.PythonSystemInstance();

                PyObjectPtr py_image = MakePyObjectPtr(tensor_converter.ConvertPy(cmd_list, image));
                gpu_system.ExecuteAndReset(cmd_list);

                auto py_upsampled_image = python_system.CallObject(
                    *super_res_process_method_, py_image, image.Width(0), image.Height(0), FormatChannels(image.Format()), false);

                tensor_converter.ConvertPy(cmd_list, *py_upsampled_image, upsampled_residual_tex, GpuFormat::RGBA8_UNorm,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess, "upsampled_residual_tex");
                gpu_system.ExecuteAndReset(cmd_list);
            }

            GpuTexture2D upsampled_tex;
            {
                constexpr uint32_t BlockDim = 16;

                PerfRegion perf(aihi_.PerfProfilerInstance(), "Blend", &cmd_list);

                upsampled_tex =
                    GpuTexture2D(gpu_system, upsampled_residual_tex.Width(0), upsampled_residual_tex.Height(0), 1, image.Format(),
                        GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable, "upsampled_tex");

                GpuConstantBufferOfType<BlendConstantBuffer> blend_cb(gpu_system, "blend_cb");
                blend_cb->dest_size = {upsampled_tex.Width(0), upsampled_tex.Height(0)};
                blend_cb->scale = upsampled_tex.Width(0) / image.Width(0);
                blend_cb.UploadStaging();
                const GpuConstantBufferView blend_cbv(gpu_system, blend_cb);

                const GpuShaderResourceView upsampled_residual_srv(gpu_system, upsampled_residual_tex);
                const GpuShaderResourceView input_srv(gpu_system, image, ToLinearFormat(image.Format()));
                GpuUnorderedAccessView upsampled_net_uav(gpu_system, upsampled_tex, ToLinearFormat(upsampled_tex.Format()));

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &blend_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"upsampled_residual_tex", &upsampled_residual_srv},
                    {"input_tex", &input_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"upsampled_net_tex", &upsampled_net_uav},
                };
                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(blend_pipeline_, {DivUp(upsampled_tex.Width(0), BlockDim), DivUp(upsampled_tex.Height(0), BlockDim), 1},
                    shader_binding);
            }

            return upsampled_tex;
        }

        GpuTexture2D Resize(GpuCommandList& cmd_list, const GpuTexture2D& upsampled_tex, uint32_t target_width, uint32_t target_height)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            PerfRegion perf(aihi_.PerfProfilerInstance(), "Resize", &cmd_list);

            GpuTexture2D resized_x_tex(gpu_system, target_width, upsampled_tex.Height(0), 1, upsampled_tex.Format(),
                GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess, "resize_x_tex");
            GpuTexture2D resized_image(gpu_system, target_width, target_height, 1, upsampled_tex.Format(),
                GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable, "resized_image");

            constexpr uint32_t BlockDim = 16;

            {
                GpuConstantBufferOfType<ResizeConstantBuffer> resize_x_cb(gpu_system, "resize_x_cb");
                resize_x_cb->src_roi = {0, 0, upsampled_tex.Width(0), upsampled_tex.Height(0)};
                resize_x_cb->dest_size = {target_width, upsampled_tex.Height(0)};
                resize_x_cb->scale = static_cast<float>(upsampled_tex.Width(0)) / target_width;
                resize_x_cb->x_dir = true;
                resize_x_cb.UploadStaging();
                const GpuConstantBufferView resize_x_cbv(gpu_system, resize_x_cb);

                const GpuShaderResourceView input_srv(gpu_system, upsampled_tex);
                GpuUnorderedAccessView output_uav(gpu_system, resized_x_tex, ToLinearFormat(resized_x_tex.Format()));

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &resize_x_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"input_tex", &input_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"output_tex", &output_uav},
                };
                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(
                    resize_pipeline_, {DivUp(target_width, BlockDim), DivUp(upsampled_tex.Height(0), BlockDim), 1}, shader_binding);
            }
            {
                const GpuShaderResourceView input_srv(gpu_system, resized_x_tex);
                GpuUnorderedAccessView output_uav(gpu_system, resized_image, ToLinearFormat(resized_image.Format()));

                GpuConstantBufferOfType<ResizeConstantBuffer> resize_y_cb(gpu_system, "resize_y_cb");
                resize_y_cb->src_roi = {0, 0, target_width, upsampled_tex.Height(0)};
                resize_y_cb->dest_size = {target_width, target_height};
                resize_y_cb->scale = static_cast<float>(upsampled_tex.Height(0)) / target_height;
                resize_y_cb->x_dir = false;
                resize_y_cb.UploadStaging();
                const GpuConstantBufferView resize_y_cbv(gpu_system, resize_y_cb);

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &resize_y_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"input_tex", &input_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"output_tex", &output_uav},
                };
                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(resize_pipeline_, {DivUp(target_width, BlockDim), DivUp(target_height, BlockDim), 1}, shader_binding);
            }

            return resized_image;
        }

    private:
        AIHoloImagerInternal& aihi_;

        PyObjectPtr super_res_module_;
        PyObjectPtr super_res_class_;
        PyObjectPtr super_res_;
        PyObjectPtr super_res_process_method_;
        std::future<void> py_init_future_;
        bool py_init_finished_ = false;

        struct BlendConstantBuffer
        {
            glm::uvec2 dest_size;
            uint32_t scale;
            uint32_t padding;
        };
        GpuComputePipeline blend_pipeline_;

        struct ResizeConstantBuffer
        {
            glm::uvec4 src_roi;
            glm::uvec2 dest_size;
            float scale;
            uint32_t x_dir;
        };
        GpuComputePipeline resize_pipeline_;
    };

    SuperResolution::SuperResolution() noexcept = default;
    SuperResolution::SuperResolution(AIHoloImagerInternal& aihi) : impl_(std::make_unique<Impl>(aihi))
    {
    }
    SuperResolution::~SuperResolution() noexcept = default;

    SuperResolution::SuperResolution(SuperResolution&& other) noexcept = default;
    SuperResolution& SuperResolution::operator=(SuperResolution&& other) noexcept = default;

    AIHoloImagerInternal::ProjectionDesc SuperResolution::Process(const AIHoloImagerInternal::ProjectionDesc& input, float scale)
    {
        return impl_->Process(input, scale);
    }
} // namespace AIHoloImager
