// Copyright (c) 2024 Minmin Gong
//

#include "MultiViewRenderer.hpp"

#include <format>
#include <numbers>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "Gpu/GpuBufferHelper.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuDescriptorAllocator.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuSampler.hpp"
#include "Gpu/GpuShader.hpp"
#include "Gpu/GpuTexture.hpp"
#include "MvDiffusion/MultiViewDiffusion.hpp"

#include "CompiledShader/MvRenderer/BlendCs.h"
#include "CompiledShader/MvRenderer/CalcDiffusionBoxCs.h"
#include "CompiledShader/MvRenderer/CalcRenderedBoxCs.h"
#include "CompiledShader/MvRenderer/DownsampleCs.h"
#include "CompiledShader/MvRenderer/RenderPs.h"
#include "CompiledShader/MvRenderer/RenderVs.h"

using namespace AIHoloImager;

namespace
{
    // The angles are defined by zero123plus v1.2 (https://github.com/SUDO-AI-3D/zero123plus)
    constexpr float Azimuths[] = {30, 90, 150, 210, 270, 330};
    constexpr float Elevations[] = {20, -10, 20, -10, 20, -10};
    constexpr float Fov = std::numbers::pi_v<float> / 6;
    const float MvScale = 1.6f; // The fine-tuned zero123plus in InstantMesh has a scale
                                // (https://github.com/TencentARC/InstantMesh/commit/34c193cc96eebd46deb7c48a76613753ad777122)

    glm::vec3 SphericalCameraPose(float azimuth_deg, float elevation_deg, float radius)
    {
        const float azimuth = glm::radians(azimuth_deg);
        const float elevation = glm::radians(elevation_deg);

        const float sin_azimuth = -std::sin(azimuth);
        const float cos_azimuth = std::cos(azimuth);

        const float sin_elevation = std::sin(elevation);
        const float cos_elevation = std::cos(elevation);

        const float x = cos_elevation * cos_azimuth;
        const float y = sin_elevation;
        const float z = cos_elevation * sin_azimuth;
        return glm::vec3(x, y, z) * radius;
    }
} // namespace

namespace AIHoloImager
{
    class MultiViewRenderer::Impl
    {
    public:
        Impl(GpuSystem& gpu_system, PythonSystem& python_system, uint32_t width, uint32_t height)
            : gpu_system_(gpu_system), python_system_(python_system), proj_mtx_(glm::perspectiveRH_ZO(Fov, 1.0f, 0.1f, 30.0f))
        {
            constexpr DXGI_FORMAT ColorFmt = DXGI_FORMAT_R8G8B8A8_UNORM;
            constexpr DXGI_FORMAT DsFmt = DXGI_FORMAT_D32_FLOAT;

            ssaa_rt_tex_ = GpuTexture2D(gpu_system_, width * SsaaScale, height * SsaaScale, 1, ColorFmt,
                GpuResourceFlag::RenderTarget | GpuResourceFlag::UnorderedAccess, L"ssaa_rt_tex_");
            ssaa_rtv_ = GpuRenderTargetView(gpu_system_, ssaa_rt_tex_);
            ssaa_srv_ = GpuShaderResourceView(gpu_system_, ssaa_rt_tex_);
            ssaa_uav_ = GpuUnorderedAccessView(gpu_system_, ssaa_rt_tex_);

            ssaa_ds_tex_ =
                GpuTexture2D(gpu_system_, width * SsaaScale, height * SsaaScale, 1, DsFmt, GpuResourceFlag::DepthStencil, L"ssaa_ds_tex_");
            ssaa_dsv_ = GpuDepthStencilView(gpu_system_, ssaa_ds_tex_);

            init_view_tex_ = GpuTexture2D(gpu_system_, width, height, 1, ColorFmt, GpuResourceFlag::UnorderedAccess, L"init_view_tex_");
            init_view_uav_ = GpuUnorderedAccessView(gpu_system_, init_view_tex_);
            for (size_t i = 0; i < std::size(multi_view_texs_); ++i)
            {
                multi_view_texs_[i] = GpuTexture2D(gpu_system_, width, height, 1, init_view_tex_.Format(), GpuResourceFlag::UnorderedAccess,
                    std::format(L"multi_view_tex_{}", i));
                multi_view_uavs_[i] = GpuUnorderedAccessView(gpu_system_, multi_view_texs_[i]);
            }

            {
                render_cb_ = ConstantBuffer<RenderConstantBuffer>(gpu_system_, 1, L"render_cb_");

                const ShaderInfo shaders[] = {
                    {RenderVs_shader, 1, 0, 0},
                    {RenderPs_shader, 0, 1, 0},
                };

                const DXGI_FORMAT rtv_formats[] = {ssaa_rt_tex_.Format()};

                GpuRenderPipeline::States states;
                states.cull_mode = GpuRenderPipeline::CullMode::ClockWise;
                states.depth_enable = true;
                states.rtv_formats = rtv_formats;
                states.dsv_format = ssaa_ds_tex_.Format();

                const GpuStaticSampler point_sampler(GpuStaticSampler::Filter::Point, GpuStaticSampler::AddressMode::Clamp);

                const GpuVertexAttribs vertex_attribs(std::span<const GpuVertexAttrib>({
                    {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT},
                    {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT},
                }));

                render_pipeline_ = GpuRenderPipeline(gpu_system_, shaders, vertex_attribs, std::span(&point_sampler, 1), states);
            }

            {
                const ShaderInfo shader = {DownsampleCs_shader, 0, 1, 1};
                downsample_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const ShaderInfo shader = {CalcRenderedBoxCs_shader, 0, 1, 1};
                calc_rendered_box_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                calc_diffusion_box_cb_ = ConstantBuffer<CalcDiffusionBoxConstantBuffer>(gpu_system_, 1, L"calc_diffusion_box_cb_");

                const ShaderInfo shader = {CalcDiffusionBoxCs_shader, 1, 1, 1};
                calc_diffusion_box_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }

            {
                blend_cb_ = ConstantBuffer<BlendConstantBuffer>(gpu_system_, 1, L"blend_cb_");

                const GpuStaticSampler bilinear_sampler(
                    {GpuStaticSampler::Filter::Linear, GpuStaticSampler::Filter::Linear}, GpuStaticSampler::AddressMode::Clamp);

                const ShaderInfo shader = {BlendCs_shader, 1, 2, 1};
                blend_pipeline_ = GpuComputePipeline(gpu_system_, shader, std::span{&bilinear_sampler, 1});
            }
            bb_tex_ = GpuTexture2D(gpu_system_, 4, 2, 1, DXGI_FORMAT_R32_UINT, GpuResourceFlag::UnorderedAccess, L"bb_tex_");
            bb_srv_ = GpuShaderResourceView(gpu_system_, bb_tex_);
            bb_uav_ = GpuUnorderedAccessView(gpu_system_, bb_tex_);
        }

        Result Render(const Mesh& mesh, [[maybe_unused]] const std::filesystem::path& tmp_dir)
        {
            assert(mesh.MeshVertexDesc().Stride() == sizeof(VertexFormat));

            GpuBuffer vb(gpu_system_, static_cast<uint32_t>(mesh.VertexBuffer().size() * sizeof(float)), GpuHeap::Upload,
                GpuResourceFlag::None, L"vb");
            memcpy(vb.Map(), mesh.VertexBuffer().data(), vb.Size());
            vb.Unmap(GpuRange{0, vb.Size()});

            GpuBuffer ib(gpu_system_, static_cast<uint32_t>(mesh.IndexBuffer().size() * sizeof(uint32_t)), GpuHeap::Upload,
                GpuResourceFlag::None, L"ib");
            memcpy(ib.Map(), mesh.IndexBuffer().data(), ib.Size());
            ib.Unmap(GpuRange{0, ib.Size()});

            GpuTexture2D albedo_gpu_tex;
            {
                const auto& albedo_tex = mesh.AlbedoTexture();
                albedo_gpu_tex = GpuTexture2D(gpu_system_, albedo_tex.Width(), albedo_tex.Height(), 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                    GpuResourceFlag::None, L"albedo_gpu_tex");
                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                albedo_gpu_tex.Upload(gpu_system_, cmd_list, 0, albedo_tex.Data());
                gpu_system_.Execute(std::move(cmd_list));
            }
            GpuShaderResourceView albedo_gpu_srv(gpu_system_, albedo_gpu_tex);

            constexpr float CameraDist = 10;

#ifdef AIHI_KEEP_INTERMEDIATES
            const auto output_dir = tmp_dir / "Renderer";
            std::filesystem::create_directories(output_dir);
#endif

            const uint32_t num_indices = static_cast<uint32_t>(mesh.IndexBuffer().size());

            {
                GpuCommandList cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                RenderToSsaa(cmd_list, vb, ib, num_indices, albedo_gpu_srv, 0, 45, CameraDist);
                Downsample(cmd_list, init_view_tex_, init_view_uav_);
                gpu_system_.Execute(std::move(cmd_list));
            }

            GpuTexture2D mv_diffusion_gpu_tex;
            {
                Texture init_view_cpu_tex = Texture(init_view_tex_.Width(0), init_view_tex_.Height(0), FormatSize(init_view_tex_.Format()));
                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                init_view_tex_.Readback(gpu_system_, cmd_list, 0, init_view_cpu_tex.Data());
                gpu_system_.Execute(std::move(cmd_list));

#ifdef AIHI_KEEP_INTERMEDIATES
                SaveTexture(init_view_cpu_tex, output_dir / "InitView.png");
#endif

                Ensure3Channel(init_view_cpu_tex);
                Texture mv_diffusion_tex;
                {
                    MultiViewDiffusion mv_diffusion(python_system_);
                    mv_diffusion_tex = mv_diffusion.Generate(init_view_cpu_tex);
                }
                Ensure4Channel(mv_diffusion_tex);

#ifdef AIHI_KEEP_INTERMEDIATES
                SaveTexture(mv_diffusion_tex, output_dir / "Diffusion.png");
#endif

                cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                mv_diffusion_gpu_tex = GpuTexture2D(gpu_system_, mv_diffusion_tex.Width(), mv_diffusion_tex.Height(), 1,
                    DXGI_FORMAT_R8G8B8A8_UNORM, GpuResourceFlag::None, L"mv_diffusion_gpu_tex");
                mv_diffusion_gpu_tex.Upload(gpu_system_, cmd_list, 0, mv_diffusion_tex.Data());
                gpu_system_.Execute(std::move(cmd_list));
            }

            Result ret;
            for (size_t i = 0; i < std::size(Azimuths); ++i)
            {
                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                RenderToSsaa(cmd_list, vb, ib, num_indices, albedo_gpu_srv, Azimuths[i], Elevations[i], CameraDist, MvScale);
                BlendWithDiffusion(cmd_list, mv_diffusion_gpu_tex, static_cast<uint32_t>(i));
                Downsample(cmd_list, multi_view_texs_[i], multi_view_uavs_[i]);

                ret.multi_view_images[i] =
                    Texture(multi_view_texs_[i].Width(0), multi_view_texs_[i].Height(0), FormatSize(multi_view_texs_[i].Format()));

                multi_view_texs_[i].Readback(gpu_system_, cmd_list, 0, ret.multi_view_images[i].Data());
                gpu_system_.Execute(std::move(cmd_list));

                Ensure3Channel(ret.multi_view_images[i]);

#ifdef AIHI_KEEP_INTERMEDIATES
                SaveTexture(ret.multi_view_images[i], output_dir / std::format("View{}.png", i));
#endif
            }

            return ret;
        }

    private:
        void RenderToSsaa(GpuCommandList& cmd_list, GpuBuffer& vb, GpuBuffer& ib, uint32_t num_indices,
            const GpuShaderResourceView& albedo_srv, float camera_azimuth, float camera_elevation, float camera_dist, float scale = 1)
        {
            const glm::vec3 camera_pos = SphericalCameraPose(camera_azimuth, camera_elevation, camera_dist);
            const glm::vec3 camera_dir = -glm::normalize(camera_pos);
            glm::vec3 up_vec;
            if (-camera_dir.y > 0.95f)
            {
                up_vec = glm::vec3(1, 0, 0);
            }
            else
            {
                up_vec = glm::vec3(0, 1, 0);
            }

            const glm::mat4x4 view_mtx = glm::lookAtRH(camera_pos, glm::vec3(0, 0, 0), up_vec);

            render_cb_->mvp = glm::transpose(proj_mtx_ * view_mtx);
            render_cb_.UploadToGpu();

            const float clear_clr[] = {0, 0, 0, 0};
            cmd_list.Clear(ssaa_rtv_, clear_clr);
            cmd_list.ClearDepth(ssaa_dsv_, 1);

            const GpuCommandList::VertexBufferBinding vb_bindings[] = {{&vb, 0, sizeof(VertexFormat)}};
            const GpuCommandList::IndexBufferBinding ib_binding = {&ib, 0, DXGI_FORMAT_R32_UINT};

            const GeneralConstantBuffer* cbs[] = {&render_cb_};
            const GpuShaderResourceView* srvs[] = {&albedo_srv};
            const GpuCommandList::ShaderBinding shader_bindings[] = {
                {cbs, {}, {}},
                {{}, srvs, {}},
            };

            const GpuRenderTargetView* rtvs[] = {&ssaa_rtv_};

            const float offset_scale = (scale - 1) / 2;
            const GpuViewport viewport = {-offset_scale * ssaa_rt_tex_.Width(0), -offset_scale * ssaa_rt_tex_.Height(0),
                scale * ssaa_rt_tex_.Width(0), scale * ssaa_rt_tex_.Height(0)};
            const GpuRect scissor_rc = {0, 0, static_cast<int32_t>(ssaa_rt_tex_.Width(0)), static_cast<int32_t>(ssaa_rt_tex_.Height(0))};

            cmd_list.Render(render_pipeline_, vb_bindings, &ib_binding, num_indices, shader_bindings, rtvs, &ssaa_dsv_,
                std::span(&viewport, 1), std::span(&scissor_rc, 1));
        }

        void Downsample(GpuCommandList& cmd_list, GpuTexture2D& target_tex, GpuUnorderedAccessView& target_uav)
        {
            constexpr uint32_t BlockDim = 16;

            const GpuShaderResourceView* srvs[] = {&ssaa_srv_};
            GpuUnorderedAccessView* uavs[] = {&target_uav};
            const GpuCommandList::ShaderBinding shader_binding = {{}, srvs, uavs};
            cmd_list.Compute(
                downsample_pipeline_, DivUp(target_tex.Width(0), BlockDim), DivUp(target_tex.Height(0), BlockDim), 1, shader_binding);
        }

        void BlendWithDiffusion(GpuCommandList& cmd_list, GpuTexture2D& mv_diffusion_tex, uint32_t index)
        {
            constexpr uint32_t BlockDim = 16;
            constexpr uint32_t AtlasWidth = 2;
            constexpr uint32_t AtlasHeight = 3;

            const uint32_t view_width = mv_diffusion_tex.Width(0) / AtlasWidth;
            const uint32_t view_height = mv_diffusion_tex.Height(0) / AtlasHeight;

            const uint32_t atlas_y = index / AtlasWidth;
            const uint32_t atlas_x = index - atlas_y * AtlasWidth;
            const uint32_t atlas_offset_x = atlas_x * view_width;
            const uint32_t atlas_offset_y = atlas_y * view_height;

            uint32_t bb_init[] = {ssaa_rt_tex_.Width(0), ssaa_rt_tex_.Height(0), 0, 0, view_width, view_height, 0, 0};
            bb_tex_.Upload(gpu_system_, cmd_list, 0, bb_init);

            GpuShaderResourceView mv_diffusion_srv(gpu_system_, mv_diffusion_tex);

            {
                const GpuShaderResourceView* srvs[] = {&ssaa_srv_};
                GpuUnorderedAccessView* uavs[] = {&bb_uav_};
                const GpuCommandList::ShaderBinding shader_binding = {{}, srvs, uavs};
                cmd_list.Compute(calc_rendered_box_pipeline_, DivUp(ssaa_rt_tex_.Width(0), BlockDim),
                    DivUp(ssaa_rt_tex_.Height(0), BlockDim), 1, shader_binding);
            }
            {
                calc_diffusion_box_cb_->atlas_offset_view_size.x = atlas_offset_x;
                calc_diffusion_box_cb_->atlas_offset_view_size.y = atlas_offset_y;
                calc_diffusion_box_cb_->atlas_offset_view_size.z = view_width;
                calc_diffusion_box_cb_->atlas_offset_view_size.w = view_height;
                calc_diffusion_box_cb_.UploadToGpu();

                const GeneralConstantBuffer* cbs[] = {&calc_diffusion_box_cb_};
                const GpuShaderResourceView* srvs[] = {&mv_diffusion_srv};
                GpuUnorderedAccessView* uavs[] = {&bb_uav_};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(
                    calc_diffusion_box_pipeline_, DivUp(view_width, BlockDim), DivUp(view_height, BlockDim), 1, shader_binding);
            }
            {
                blend_cb_->atlas_offset_view_size.x = atlas_offset_x;
                blend_cb_->atlas_offset_view_size.y = atlas_offset_y;
                blend_cb_->atlas_offset_view_size.z = view_width;
                blend_cb_->atlas_offset_view_size.w = view_height;
                blend_cb_->rendered_diffusion_center.x = ssaa_rt_tex_.Width(0) / 2;
                blend_cb_->rendered_diffusion_center.y = ssaa_rt_tex_.Height(0) / 2;
                blend_cb_->rendered_diffusion_center.z = view_width / 2;
                blend_cb_->rendered_diffusion_center.w = view_height / 2;
                blend_cb_->diffusion_inv_size.x = 1.0f / mv_diffusion_tex.Width(0);
                blend_cb_->diffusion_inv_size.y = 1.0f / mv_diffusion_tex.Height(0);
                blend_cb_.UploadToGpu();

                const GeneralConstantBuffer* cbs[] = {&blend_cb_};
                const GpuShaderResourceView* srvs[] = {&mv_diffusion_srv, &bb_srv_};
                GpuUnorderedAccessView* uavs[] = {&ssaa_uav_};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(
                    blend_pipeline_, DivUp(ssaa_rt_tex_.Width(0), BlockDim), DivUp(ssaa_rt_tex_.Height(0), BlockDim), 1, shader_binding);
            }
        }

    private:
        GpuSystem& gpu_system_;
        PythonSystem& python_system_;

        GpuTexture2D init_view_tex_;
        GpuUnorderedAccessView init_view_uav_;
        GpuTexture2D multi_view_texs_[6];
        GpuUnorderedAccessView multi_view_uavs_[6];

        static constexpr uint32_t SsaaScale = 4;

        GpuTexture2D ssaa_rt_tex_;
        GpuRenderTargetView ssaa_rtv_;
        GpuShaderResourceView ssaa_srv_;
        GpuUnorderedAccessView ssaa_uav_;

        GpuTexture2D ssaa_ds_tex_;
        GpuDepthStencilView ssaa_dsv_;

        glm::mat4x4 proj_mtx_;

        struct VertexFormat
        {
            glm::vec3 pos;
            glm::vec2 texcoord;
        };
        struct RenderConstantBuffer
        {
            glm::mat4x4 mvp;
        };
        ConstantBuffer<RenderConstantBuffer> render_cb_;
        GpuRenderPipeline render_pipeline_;

        GpuComputePipeline downsample_pipeline_;

        GpuComputePipeline calc_rendered_box_pipeline_;

        struct CalcDiffusionBoxConstantBuffer
        {
            glm::uvec4 atlas_offset_view_size;
        };
        ConstantBuffer<CalcDiffusionBoxConstantBuffer> calc_diffusion_box_cb_;
        GpuComputePipeline calc_diffusion_box_pipeline_;

        struct BlendConstantBuffer
        {
            glm::uvec4 atlas_offset_view_size;
            glm::uvec4 rendered_diffusion_center;
            glm::vec4 diffusion_inv_size;
        };
        ConstantBuffer<BlendConstantBuffer> blend_cb_;
        GpuComputePipeline blend_pipeline_;
        GpuTexture2D bb_tex_;
        GpuShaderResourceView bb_srv_;
        GpuUnorderedAccessView bb_uav_;
    };

    MultiViewRenderer::MultiViewRenderer(GpuSystem& gpu_system, PythonSystem& python_system, uint32_t width, uint32_t height)
        : impl_(std::make_unique<Impl>(gpu_system, python_system, width, height))
    {
    }

    MultiViewRenderer::~MultiViewRenderer() noexcept = default;

    MultiViewRenderer::MultiViewRenderer(MultiViewRenderer&& other) noexcept = default;
    MultiViewRenderer& MultiViewRenderer::operator=(MultiViewRenderer&& other) noexcept = default;

    MultiViewRenderer::Result MultiViewRenderer::Render(const Mesh& mesh, const std::filesystem::path& tmp_dir)
    {
        return impl_->Render(mesh, tmp_dir);
    }
} // namespace AIHoloImager
