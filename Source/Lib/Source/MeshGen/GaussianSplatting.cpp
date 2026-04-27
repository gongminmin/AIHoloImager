// Copyright (c) 2026 Minmin Gong
//

#include "GaussianSplatting.hpp"

#ifdef AIHI_KEEP_INTERMEDIATES
    #include <fstream>
#endif

#include "AIHoloImager/Mesh.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Util/PerfProfiler.hpp"

#include "Sorter.hpp"

#include "CompiledShader/MeshGen/GaussianSplatting/BlendCs.h"
#include "CompiledShader/MeshGen/GaussianSplatting/PreprocessCs.h"
#include "CompiledShader/MeshGen/GaussianSplatting/RenderGs.h"
#include "CompiledShader/MeshGen/GaussianSplatting/RenderPs.h"
#include "CompiledShader/MeshGen/GaussianSplatting/RenderVs.h"

namespace
{
    constexpr uint32_t NumCoefficientsFromShDegrees(uint32_t degrees) noexcept
    {
        return (degrees + 1) * (degrees + 1);
    }
} // namespace

namespace AIHoloImager
{
    class GaussianSplatting::Impl
    {
    public:
        explicit Impl(AIHoloImagerInternal& aihi) : aihi_(aihi), sorter_(aihi)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            {
                const ShaderInfo shader = {DEFINE_SHADER(PreprocessCs)};
                preprocess_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shaders[] = {
                    {DEFINE_SHADER(RenderVs)},
                    {DEFINE_SHADER(RenderPs)},
                    {DEFINE_SHADER(RenderGs)},
                };

                GpuRenderPipeline::States states;
                states.cull_mode = GpuRenderPipeline::CullMode::CounterClockWise;

                const GpuFormat rtv_formats[] = {GSplatFmt};
                states.rtv_formats = rtv_formats;

                const GpuRenderPipeline::RenderTargetBlendDesc blend_desc{
                    .blend_enable = true,
                    .src_color_blend_factor = GpuRenderPipeline::BlendFactor::DstAlpha,
                    .dst_color_blend_factor = GpuRenderPipeline::BlendFactor::One,
                    .src_alpha_blend_factor = GpuRenderPipeline::BlendFactor::Zero,
                    .dst_alpha_blend_factor = GpuRenderPipeline::BlendFactor::InvSrcAlpha,
                };
                states.blend_states.render_targets = std::span(&blend_desc, 1);

                const GpuVertexLayout vertex_layout(gpu_system, std::span<const GpuVertexAttrib>({
                                                                    {"TEXCOORD", 0, GpuFormat::RGBA32_Float, 0},
                                                                    {"TEXCOORD", 1, GpuFormat::RGB32_Float, 1},
                                                                    {"TEXCOORD", 2, GpuFormat::RGBA32_Float, 2},
                                                                }));

                render_pipeline_ =
                    GpuRenderPipeline(gpu_system, GpuRenderPipeline::PrimitiveTopology::PointList, shaders, vertex_layout, {}, states);
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(BlendCs)};
                blend_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
        }

        void Render(const Gaussians& gaussians, const glm::mat4x4& view_mtx, const glm::mat4x4& proj_mtx, float kernel_size,
            GpuTexture2D& rendered_image)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            const uint32_t width = rendered_image.Width(0);
            const uint32_t height = rendered_image.Height(0);
            const float tan_fov_x = 1 / proj_mtx[0][0];
            const float tan_fov_y = 1 / proj_mtx[1][1];
            const float focal_x = width * proj_mtx[0][0] / 2;
            const float focal_y = height * proj_mtx[1][1] / 2;

            auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Render);

            PerfRegion gsplat_perf(aihi_.PerfProfilerInstance(), "GaussianSplatting Render", &cmd_list);

            if (intermediate_cache_.num_gaussians_allocated < gaussians.num_gaussians)
            {
                intermediate_cache_.screen_pos_extents = GpuBuffer(gpu_system, gaussians.num_gaussians * sizeof(glm::vec4),
                    GpuHeap::Default, GpuResourceFlag::UnorderedAccess, "gsplat.intermediate_cache_.screen_pos");
                intermediate_cache_.conic_opacity = GpuBuffer(gpu_system, gaussians.num_gaussians * sizeof(glm::vec4), GpuHeap::Default,
                    GpuResourceFlag::UnorderedAccess, "gsplat.intermediate_cache_.conic_opacity");
                intermediate_cache_.color = GpuBuffer(gpu_system, gaussians.num_gaussians * sizeof(glm::vec3), GpuHeap::Default,
                    GpuResourceFlag::UnorderedAccess, "gsplat.intermediate_cache_.color");
                intermediate_cache_.num_visible_gaussians_indirect_args = GpuBuffer(gpu_system, sizeof(GpuRenderIndexedArguments),
                    GpuHeap::Default, GpuResourceFlag::UnorderedAccess, "gsplat.intermediate_cache_.num_visible_gaussians_indirect_args");
                intermediate_cache_.visible_key = GpuBuffer(gpu_system, gaussians.num_gaussians * sizeof(uint32_t), GpuHeap::Default,
                    GpuResourceFlag::UnorderedAccess, "gsplat.intermediate_cache_.visible_key");
                intermediate_cache_.visible_id = GpuBuffer(gpu_system, gaussians.num_gaussians * sizeof(uint32_t), GpuHeap::Default,
                    GpuResourceFlag::UnorderedAccess, "gsplat.intermediate_cache_.visible_id");

                intermediate_cache_.num_gaussians_allocated = gaussians.num_gaussians;
            }
            if ((intermediate_cache_.gsplat_image.Width(0) != width) || (intermediate_cache_.gsplat_image.Height(0) != height))
            {
                intermediate_cache_.gsplat_image = GpuTexture2D(
                    gpu_system, width, height, 1, GSplatFmt, GpuResourceFlag::RenderTarget, "gsplat.intermediate_cache_.gsplat_image");
            }

            {
                constexpr uint32_t BlockDim = 256;

                GpuConstantBufferOfType<PreprocessConstantBuffer> preprocess_cb(gpu_system, "preprocess_cb");
                preprocess_cb->num_gaussians = gaussians.num_gaussians;
                preprocess_cb->sh_degrees = gaussians.sh_degrees;
                preprocess_cb->num_coeffs = NumCoefficientsFromShDegrees(gaussians.sh_degrees);
                preprocess_cb->kernel_size = kernel_size;
                preprocess_cb->view_mtx = glm::transpose(view_mtx);
                preprocess_cb->view_proj_mtx = glm::transpose(proj_mtx * view_mtx);
                preprocess_cb->focal = glm::vec2(focal_x, focal_y);
                preprocess_cb->tan_fov = glm::vec2(tan_fov_x, tan_fov_y);
                preprocess_cb->width_height = glm::uvec2(width, height);
                preprocess_cb.UploadStaging();
                const GpuConstantBufferView preprocess_cbv(gpu_system, preprocess_cb);

                const GpuShaderResourceView pos_srv(gpu_system, gaussians.positions, GpuFormat::RGB32_Float);
                const GpuShaderResourceView scale_srv(gpu_system, gaussians.scales, GpuFormat::RGB32_Float);
                const GpuShaderResourceView rotation_srv(gpu_system, gaussians.rotations, GpuFormat::RGBA32_Float);
                const GpuShaderResourceView sh_srv(gpu_system, gaussians.shs, GpuFormat::RGB32_Float);
                const GpuShaderResourceView opacity_srv(gpu_system, gaussians.opacities, GpuFormat::R32_Float);

                GpuUnorderedAccessView screen_pos_extents_uav(gpu_system, intermediate_cache_.screen_pos_extents, GpuFormat::RGBA32_Float);
                GpuUnorderedAccessView color_uav(gpu_system, intermediate_cache_.color, GpuFormat::R32_Float);
                GpuUnorderedAccessView conic_opacity_uav(gpu_system, intermediate_cache_.conic_opacity, GpuFormat::RGBA32_Float);
                GpuUnorderedAccessView num_visible_gaussians_uav(
                    gpu_system, intermediate_cache_.num_visible_gaussians_indirect_args, GpuFormat::R32_Uint);
                GpuUnorderedAccessView visible_key_uav(gpu_system, intermediate_cache_.visible_key, GpuFormat::R32_Uint);
                GpuUnorderedAccessView visible_id_uav(gpu_system, intermediate_cache_.visible_id, GpuFormat::R32_Uint);

                {
                    const uint32_t clear_value[] = {0, 0, 0, 0};
                    cmd_list.Clear(num_visible_gaussians_uav, clear_value);
                }
                {
                    const uint32_t clear_value[] = {0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU};
                    cmd_list.Clear(visible_key_uav, clear_value);
                }

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &preprocess_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"pos_buff", &pos_srv},
                    {"scale_buff", &scale_srv},
                    {"rotation_buff", &rotation_srv},
                    {"sh_buff", &sh_srv},
                    {"opacity_buff", &opacity_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"color_buff", &color_uav},
                    {"conic_opacity_buff", &conic_opacity_uav},
                    {"screen_pos_extents_buff", &screen_pos_extents_uav},
                    {"num_visible_gaussians_buff", &num_visible_gaussians_uav},
                    {"visible_key_buff", &visible_key_uav},
                    {"visible_id_buff", &visible_id_uav},
                };
                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(preprocess_pipeline_, {DivUp(gaussians.num_gaussians, BlockDim), 1, 1}, shader_binding);
            }

            {
                sorter_.RadixSort(cmd_list, intermediate_cache_.visible_key, GpuFormat::R32_Uint, intermediate_cache_.visible_id,
                    GpuFormat::R32_Uint, gaussians.num_gaussians, intermediate_cache_.visible_key, intermediate_cache_.visible_id, 32);
            }

            {
                GpuConstantBufferOfType<RenderConstantBuffer> render_cb(gpu_system, "render_cb");
                render_cb->width_height = glm::uvec2(width, height);
                render_cb.UploadStaging();
                const GpuConstantBufferView render_cbv(gpu_system, render_cb);

                GpuRenderTargetView gsplat_image_rtv(gpu_system, intermediate_cache_.gsplat_image);

                {
                    const float init_value[] = {0, 0, 0, 1};
                    cmd_list.Clear(gsplat_image_rtv, init_value);
                }

                const GpuCommandList::VertexBufferBinding vb_bindings[] = {
                    {&intermediate_cache_.screen_pos_extents, 0},
                    {&intermediate_cache_.color, 0},
                    {&intermediate_cache_.conic_opacity, 0},
                };
                const GpuCommandList::IndexBufferBinding ib_binding = {&intermediate_cache_.visible_id, 0, GpuFormat::R32_Uint};

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &render_cbv},
                };
                const GpuCommandList::ShaderBinding shader_bindings[] = {
                    {cbvs, {}, {}},
                    {{}, {}, {}},
                    {{}, {}, {}},
                };

                GpuRenderTargetView* rtvs[] = {&gsplat_image_rtv};

                const GpuViewport viewport = {0, 0, static_cast<float>(width), static_cast<float>(height)};
                cmd_list.RenderIndexedIndirect(render_pipeline_, vb_bindings, ib_binding,
                    intermediate_cache_.num_visible_gaussians_indirect_args, shader_bindings, rtvs, nullptr, std::span(&viewport, 1), {});
            }

            {
                constexpr uint32_t BlockDim = 16;

                GpuConstantBufferOfType<BlendConstantBuffer> blend_cb(gpu_system, "blend_cb");
                blend_cb->width_height = glm::uvec2(width, height);
                blend_cb.UploadStaging();
                const GpuConstantBufferView blend_cbv(gpu_system, blend_cb);

                const GpuShaderResourceView gsplat_srv(gpu_system, intermediate_cache_.gsplat_image, 0);

                GpuUnorderedAccessView rendered_image_uav(gpu_system, rendered_image, 0);

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &blend_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"gsplat_image", &gsplat_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"rendered_image", &rendered_image_uav},
                };
                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(blend_pipeline_, {DivUp(width, BlockDim), DivUp(height, BlockDim), 1}, shader_binding);
            }

#ifdef AIHI_KEEP_INTERMEDIATES
            rendered_image.Transition(cmd_list, GpuResourceState::Common);
#endif

            gsplat_perf.End();
            gpu_system.Execute(std::move(cmd_list));
        }

    private:
        AIHoloImagerInternal& aihi_;

        struct PreprocessConstantBuffer
        {
            uint32_t num_gaussians;
            uint32_t sh_degrees;
            uint32_t num_coeffs;
            float kernel_size;

            glm::mat4x4 view_mtx;
            glm::mat4x4 view_proj_mtx;

            glm::vec2 focal;
            glm::vec2 tan_fov;

            glm::uvec2 width_height;
            glm::uvec2 padding;
        };
        GpuComputePipeline preprocess_pipeline_;

        struct RenderConstantBuffer
        {
            glm::uvec2 width_height;
            glm::uvec2 padding;
        };
        GpuRenderPipeline render_pipeline_;

        struct BlendConstantBuffer
        {
            glm::uvec2 width_height;
            glm::uvec2 padding;
        };
        GpuComputePipeline blend_pipeline_;

        Sorter sorter_;

        struct IntermediateCache
        {
            uint32_t num_gaussians_allocated = 0;

            GpuBuffer screen_pos_extents;
            GpuBuffer conic_opacity;
            GpuBuffer color;
            GpuBuffer num_visible_gaussians_indirect_args;

            GpuBuffer visible_key;
            GpuBuffer visible_id;

            GpuTexture2D gsplat_image;
        };
        IntermediateCache intermediate_cache_;
        static constexpr GpuFormat GSplatFmt = GpuFormat::RGBA8_UNorm;
    };

    GaussianSplatting::GaussianSplatting() noexcept = default;
    GaussianSplatting::GaussianSplatting(AIHoloImagerInternal& aihi) : impl_(std::make_unique<Impl>(aihi))
    {
    }

    GaussianSplatting::~GaussianSplatting() noexcept = default;

    GaussianSplatting::GaussianSplatting(GaussianSplatting&& other) noexcept = default;
    GaussianSplatting& GaussianSplatting::operator=(GaussianSplatting&& other) noexcept = default;

    void GaussianSplatting::Render(const Gaussians& gaussians, const glm::mat4x4& view_mtx, const glm::mat4x4& proj_mtx, float kernel_size,
        GpuTexture2D& rendered_image)
    {
        impl_->Render(gaussians, view_mtx, proj_mtx, kernel_size, rendered_image);
    }

#ifdef AIHI_KEEP_INTERMEDIATES
    glm::vec3 Sh2Rgb(const glm::vec3* sh) noexcept
    {
        constexpr float C0 = 0.28209479177387814f;
        return sh[0] * C0 + 0.5f;
    }

    void SavePointCloud(GpuSystem& gpu_system, const Gaussians& gaussians, const std::filesystem::path& path)
    {
        const VertexAttrib pos_clr_vertex_attribs[] = {
            {VertexAttrib::Semantic::Position, 0, 3},
            {VertexAttrib::Semantic::Color, 0, 3},
        };
        constexpr uint32_t PosAttribIndex = 0;
        constexpr uint32_t ColorAttribIndex = 1;
        const VertexDesc pos_clr_vertex_desc(pos_clr_vertex_attribs);

        Mesh pc_mesh = Mesh(pos_clr_vertex_desc, gaussians.num_gaussians, 0);

        auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

        auto pos_rb_future = cmd_list.ReadBackAsync(gaussians.positions, [&pc_mesh, &gaussians](const void* src_data) {
            const auto* pos_data = reinterpret_cast<const glm::vec3*>(src_data);
            for (uint32_t i = 0; i < gaussians.num_gaussians; ++i)
            {
                pc_mesh.VertexData<glm::vec3>(i, PosAttribIndex) = pos_data[i];
            }
        });
        auto color_rb_future = cmd_list.ReadBackAsync(gaussians.shs, [&pc_mesh, &gaussians](const void* src_data) {
            const auto* sh_data = reinterpret_cast<const glm::vec3*>(src_data);
            const uint32_t num_coeffs = NumCoefficientsFromShDegrees(gaussians.sh_degrees);
            for (uint32_t i = 0; i < gaussians.num_gaussians; ++i)
            {
                pc_mesh.VertexData<glm::vec3>(i, ColorAttribIndex) =
                    glm::clamp(Sh2Rgb(&sh_data[i * num_coeffs]), glm::vec3(0, 0, 0), glm::vec3(1, 1, 1));
            }
        });

        gpu_system.Execute(std::move(cmd_list));

        pos_rb_future.wait();
        color_rb_future.wait();

        SaveMesh(pc_mesh, path);
    }

    Gaussians LoadGaussians(GpuSystem& gpu_system, const std::filesystem::path& path)
    {
        std::ifstream ifs(path, std::ios_base::binary | std::ios_base::in);

        Gaussians ret;

        ifs.read(reinterpret_cast<char*>(&ret.num_gaussians), sizeof(ret.num_gaussians));
        ifs.read(reinterpret_cast<char*>(&ret.sh_degrees), sizeof(ret.sh_degrees));

        const uint32_t num_coeffs = NumCoefficientsFromShDegrees(ret.sh_degrees);

        ret.positions =
            GpuBuffer(gpu_system, ret.num_gaussians * sizeof(glm::vec3), GpuHeap::Default, GpuResourceFlag::None, "gaussians.positions");
        ret.scales =
            GpuBuffer(gpu_system, ret.num_gaussians * sizeof(glm::vec3), GpuHeap::Default, GpuResourceFlag::None, "gaussians.scales");
        ret.rotations =
            GpuBuffer(gpu_system, ret.num_gaussians * sizeof(glm::vec4), GpuHeap::Default, GpuResourceFlag::None, "gaussians.rotations");
        ret.shs = GpuBuffer(
            gpu_system, ret.num_gaussians * num_coeffs * sizeof(glm::vec3), GpuHeap::Default, GpuResourceFlag::None, "gaussians.shs");
        ret.opacities =
            GpuBuffer(gpu_system, ret.num_gaussians * sizeof(float), GpuHeap::Default, GpuResourceFlag::None, "gaussians.opacities");

        auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

        cmd_list.Upload(ret.positions,
            [&ifs, &ret](void* dst_data) { ifs.read(reinterpret_cast<char*>(dst_data), ret.num_gaussians * sizeof(glm::vec3)); });
        cmd_list.Upload(ret.scales,
            [&ifs, &ret](void* dst_data) { ifs.read(reinterpret_cast<char*>(dst_data), ret.num_gaussians * sizeof(glm::vec3)); });
        cmd_list.Upload(ret.rotations,
            [&ifs, &ret](void* dst_data) { ifs.read(reinterpret_cast<char*>(dst_data), ret.num_gaussians * sizeof(glm::vec4)); });
        cmd_list.Upload(ret.shs, [&ifs, &ret, num_coeffs](void* dst_data) {
            ifs.read(reinterpret_cast<char*>(dst_data), ret.num_gaussians * num_coeffs * sizeof(glm::vec3));
        });
        cmd_list.Upload(ret.opacities,
            [&ifs, &ret](void* dst_data) { ifs.read(reinterpret_cast<char*>(dst_data), ret.num_gaussians * sizeof(float)); });

        gpu_system.Execute(std::move(cmd_list));

        return ret;
    }

    void SaveGaussians(GpuSystem& gpu_system, const Gaussians& gaussians, const std::filesystem::path& path)
    {
        std::ofstream ofs(path, std::ios_base::binary | std::ios_base::out);

        ofs.write(reinterpret_cast<const char*>(&gaussians.num_gaussians), sizeof(gaussians.num_gaussians));
        ofs.write(reinterpret_cast<const char*>(&gaussians.sh_degrees), sizeof(gaussians.sh_degrees));

        auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

        auto pos_rb_future = cmd_list.ReadBackAsync(gaussians.positions, [&ofs, &gaussians](const void* src_data) {
            ofs.write(reinterpret_cast<const char*>(src_data), gaussians.num_gaussians * sizeof(glm::vec3));
        });
        auto scales_rb_future = cmd_list.ReadBackAsync(gaussians.scales, [&ofs, &gaussians](const void* src_data) {
            ofs.write(reinterpret_cast<const char*>(src_data), gaussians.num_gaussians * sizeof(glm::vec3));
        });
        auto rotations_rb_future = cmd_list.ReadBackAsync(gaussians.rotations, [&ofs, &gaussians](const void* src_data) {
            ofs.write(reinterpret_cast<const char*>(src_data), gaussians.num_gaussians * sizeof(glm::vec4));
        });
        auto shs_rb_future = cmd_list.ReadBackAsync(gaussians.shs, [&ofs, &gaussians](const void* src_data) {
            const uint32_t num_coeffs = NumCoefficientsFromShDegrees(gaussians.sh_degrees);
            ofs.write(reinterpret_cast<const char*>(src_data), gaussians.num_gaussians * num_coeffs * sizeof(glm::vec3));
        });
        auto opacities_rb_future = cmd_list.ReadBackAsync(gaussians.opacities, [&ofs, &gaussians](const void* src_data) {
            ofs.write(reinterpret_cast<const char*>(src_data), gaussians.num_gaussians * sizeof(float));
        });

        gpu_system.Execute(std::move(cmd_list));

        pos_rb_future.wait();
        scales_rb_future.wait();
        rotations_rb_future.wait();
        shs_rb_future.wait();
        opacities_rb_future.wait();
    }
#endif
} // namespace AIHoloImager
