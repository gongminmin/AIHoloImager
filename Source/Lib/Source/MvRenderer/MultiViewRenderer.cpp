// Copyright (c) 2024 Minmin Gong
//

#include "MultiViewRenderer.hpp"

#include <format>

#include "Gpu/GpuBufferHelper.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuDescriptorAllocator.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuShader.hpp"
#include "Gpu/GpuTexture.hpp"
#include "MvDiffusion/MultiViewDiffusion.hpp"

#include "CompiledShader/BlendCs.h"
#include "CompiledShader/CalcDiffusionBoxCs.h"
#include "CompiledShader/CalcRenderedBoxCs.h"
#include "CompiledShader/DownsampleCs.h"
#include "CompiledShader/RenderPs.h"
#include "CompiledShader/RenderVs.h"

using namespace AIHoloImager;
using namespace DirectX;

namespace
{
    // The angles are defined by zero123plus v1.2 (https://github.com/SUDO-AI-3D/zero123plus)
    constexpr float Azimuths[] = {30, 90, 150, 210, 270, 330};
    constexpr float Elevations[] = {20, -10, 20, -10, 20, -10};
    constexpr float Fov = XM_PI / 6;
    const float MvScale = 1.6f; // The fine-tuned zero123plus in InstantMesh has a scale
                                // (https://github.com/TencentARC/InstantMesh/commit/34c193cc96eebd46deb7c48a76613753ad777122)

    XMVECTOR SphericalCameraPose(float azimuth_deg, float elevation_deg, float radius)
    {
        const float azimuth = XMConvertToRadians(azimuth_deg);
        const float elevation = XMConvertToRadians(elevation_deg);

        float sin_azimuth;
        float cos_azimuth;
        XMScalarSinCos(&sin_azimuth, &cos_azimuth, azimuth);

        float sin_elevation;
        float cos_elevation;
        XMScalarSinCos(&sin_elevation, &cos_elevation, elevation);

        const float x = cos_elevation * cos_azimuth;
        const float y = sin_elevation;
        const float z = cos_elevation * sin_azimuth;
        return XMVectorSet(x, y, z, 0) * radius;
    }
} // namespace

namespace AIHoloImager
{
    class MultiViewRenderer::Impl
    {
    public:
        Impl(GpuSystem& gpu_system, PythonSystem& python_system, uint32_t width, uint32_t height)
            : gpu_system_(gpu_system), python_system_(python_system), proj_mtx_(XMMatrixPerspectiveFovLH(Fov, 1, 0.1f, 30))
        {
            rtv_desc_block_ = gpu_system_.AllocRtvDescBlock(1);
            const uint32_t rtv_descriptor_size = gpu_system_.RtvDescSize();

            dsv_desc_block_ = gpu_system_.AllocDsvDescBlock(1);
            const uint32_t dsv_descriptor_size = gpu_system_.DsvDescSize();

            constexpr DXGI_FORMAT ColorFmt = DXGI_FORMAT_R8G8B8A8_UNORM;
            constexpr DXGI_FORMAT DsFmt = DXGI_FORMAT_D32_FLOAT;

            ssaa_rt_tex_ = GpuTexture2D(gpu_system_, width * SsaaScale, height * SsaaScale, 1, ColorFmt,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON,
                L"ssaa_rt_tex_");
            ssaa_rtv_ = GpuRenderTargetView(
                gpu_system_, ssaa_rt_tex_, DXGI_FORMAT_UNKNOWN, OffsetHandle(rtv_desc_block_.CpuHandle(), 0, rtv_descriptor_size));

            ssaa_ds_tex_ = GpuTexture2D(gpu_system_, width * SsaaScale, height * SsaaScale, 1, DsFmt,
                D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_COMMON, L"ssaa_ds_tex_");
            ssaa_dsv_ = GpuDepthStencilView(
                gpu_system_, ssaa_ds_tex_, DXGI_FORMAT_UNKNOWN, OffsetHandle(dsv_desc_block_.CpuHandle(), 0, dsv_descriptor_size));

            init_view_tex_ = GpuTexture2D(gpu_system_, width, height, 1, ColorFmt, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON, L"init_view_tex_");
            for (size_t i = 0; i < std::size(multi_view_texs_); ++i)
            {
                multi_view_texs_[i] = GpuTexture2D(gpu_system_, width, height, 1, init_view_tex_.Format(),
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, std::format(L"multi_view_tex_{}", i));
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

                D3D12_STATIC_SAMPLER_DESC point_sampler_desc{};
                point_sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
                point_sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                point_sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                point_sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                point_sampler_desc.MaxAnisotropy = 16;
                point_sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
                point_sampler_desc.MinLOD = 0.0f;
                point_sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
                point_sampler_desc.ShaderRegister = 0;

                const D3D12_INPUT_ELEMENT_DESC input_elems[] = {
                    {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                    {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                };

                render_pipeline_ = GpuRenderPipeline(gpu_system_, shaders, input_elems, std::span(&point_sampler_desc, 1), states);
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

                D3D12_STATIC_SAMPLER_DESC bilinear_sampler_desc{};
                bilinear_sampler_desc.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
                bilinear_sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                bilinear_sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                bilinear_sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                bilinear_sampler_desc.MaxAnisotropy = 16;
                bilinear_sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
                bilinear_sampler_desc.MinLOD = 0.0f;
                bilinear_sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
                bilinear_sampler_desc.ShaderRegister = 0;

                const ShaderInfo shader = {BlendCs_shader, 1, 2, 1};
                blend_pipeline_ = GpuComputePipeline(gpu_system_, shader, std::span{&bilinear_sampler_desc, 1});
            }
            bb_tex_ = GpuTexture2D(gpu_system_, 4, 2, 1, DXGI_FORMAT_R32_UINT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON, L"bb_tex_");
        }

        ~Impl() noexcept
        {
            gpu_system_.DeallocDsvDescBlock(std::move(dsv_desc_block_));
            gpu_system_.DeallocRtvDescBlock(std::move(rtv_desc_block_));
        }

        Result Render(const Mesh& mesh)
        {
            GpuBuffer vb(gpu_system_, static_cast<uint32_t>(mesh.Vertices().size() * sizeof(Mesh::VertexFormat)), D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, L"vb");
            memcpy(vb.Map(), mesh.Vertices().data(), vb.Size());
            vb.Unmap(D3D12_RANGE{0, vb.Size()});

            GpuBuffer ib(gpu_system_, static_cast<uint32_t>(mesh.Indices().size() * sizeof(uint32_t)), D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, L"ib");
            memcpy(ib.Map(), mesh.Indices().data(), ib.Size());
            ib.Unmap(D3D12_RANGE{0, ib.Size()});

            GpuTexture2D albedo_gpu_tex;
            {
                const auto& albedo_tex = mesh.AlbedoTexture();
                albedo_gpu_tex = GpuTexture2D(gpu_system_, albedo_tex.Width(), albedo_tex.Height(), 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);
                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Compute);
                albedo_gpu_tex.Upload(gpu_system_, cmd_list, 0, albedo_tex.Data());
                gpu_system_.Execute(std::move(cmd_list));
            }

            constexpr float CameraDist = 10;

            const uint32_t num_indices = static_cast<uint32_t>(mesh.Indices().size());

            {
                GpuCommandList cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                RenderToSsaa(cmd_list, vb, ib, num_indices, albedo_gpu_tex, 0, 45, CameraDist);
                Downsample(cmd_list, init_view_tex_);
                gpu_system_.Execute(std::move(cmd_list));
            }

            GpuTexture2D mv_diffusion_gpu_tex;
            {
                Texture init_view_cpu_tex;
                {
                    init_view_cpu_tex = Texture(init_view_tex_.Width(0), init_view_tex_.Height(0), FormatSize(init_view_tex_.Format()));
                    auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                    init_view_tex_.Readback(gpu_system_, cmd_list, 0, init_view_cpu_tex.Data());
                    gpu_system_.Execute(std::move(cmd_list));
                }

                Ensure3Channel(init_view_cpu_tex);
                MultiViewDiffusion mv_diffusion(python_system_);
                Texture mv_diffusion_tex = mv_diffusion.Generate(init_view_cpu_tex);
                Ensure4Channel(mv_diffusion_tex);

                {
                    auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                    mv_diffusion_gpu_tex = GpuTexture2D(gpu_system_, mv_diffusion_tex.Width(), mv_diffusion_tex.Height(), 1,
                        DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, L"mv_diffusion_gpu_tex");
                    mv_diffusion_gpu_tex.Upload(gpu_system_, cmd_list, 0, mv_diffusion_tex.Data());
                    gpu_system_.Execute(std::move(cmd_list));
                }
            }

            for (size_t i = 0; i < std::size(Azimuths); ++i)
            {
                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                RenderToSsaa(cmd_list, vb, ib, num_indices, albedo_gpu_tex, Azimuths[i], Elevations[i], CameraDist, MvScale);
                BlendWithDiffusion(cmd_list, mv_diffusion_gpu_tex, static_cast<uint32_t>(i));
                Downsample(cmd_list, multi_view_texs_[i]);
                gpu_system_.Execute(std::move(cmd_list));
                gpu_system_.WaitForGpu();
            }

            mv_diffusion_gpu_tex.Reset();

            Result ret;
            for (uint32_t i = 0; i < 6; ++i)
            {
                ret.multi_view_images[i] =
                    Texture(multi_view_texs_[i].Width(0), multi_view_texs_[i].Height(0), FormatSize(multi_view_texs_[i].Format()));

                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                multi_view_texs_[i].Readback(gpu_system_, cmd_list, 0, ret.multi_view_images[i].Data());
                gpu_system_.Execute(std::move(cmd_list));

                Ensure3Channel(ret.multi_view_images[i]);
            }

            return ret;
        }

    private:
        void RenderToSsaa(GpuCommandList& cmd_list, GpuBuffer& vb, GpuBuffer& ib, uint32_t num_indices, const GpuTexture2D& albedo_tex,
            float camera_azimuth, float camera_elevation, float camera_dist, float scale = 1)
        {
            auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();

            const XMVECTOR camera_pos = SphericalCameraPose(camera_azimuth, camera_elevation, camera_dist);
            const XMVECTOR camera_dir = -XMVector3Normalize(camera_pos);
            XMVECTOR up_vec;
            if (-XMVectorGetY(camera_dir) > 0.95f)
            {
                up_vec = XMVectorSet(1, 0, 0, 0);
            }
            else
            {
                up_vec = XMVectorSet(0, 1, 0, 0);
            }

            const XMMATRIX view_mtx = XMMatrixLookAtLH(camera_pos, XMVectorSet(0, 0, 0, 1.0f), up_vec);

            XMStoreFloat4x4(&render_cb_->mvp, XMMatrixTranspose(view_mtx * proj_mtx_));
            render_cb_.UploadToGpu();

            const float clear_clr[] = {0, 0, 0, 0};
            d3d12_cmd_list->ClearRenderTargetView(ssaa_rtv_.CpuHandle(), clear_clr, 0, nullptr);
            d3d12_cmd_list->ClearDepthStencilView(
                ssaa_dsv_.CpuHandle(), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1, 0, 0, nullptr);

            const GpuCommandList::VertexBufferBinding vb_bindings[] = {{&vb, 0, sizeof(Mesh::VertexFormat)}};
            const GpuCommandList::IndexBufferBinding ib_binding = {&ib, 0, DXGI_FORMAT_R32_UINT};

            const GeneralConstantBuffer* cbs[] = {&render_cb_};
            const GpuTexture2D* srv_texs[] = {&albedo_tex};
            const GpuCommandList::ShaderBinding shader_bindings[] = {
                {cbs, {}, {}},
                {{}, srv_texs, {}},
            };

            const GpuCommandList::RenderTargetBinding rt_bindings[] = {{&ssaa_rt_tex_, &ssaa_rtv_}};
            const GpuCommandList::DepthStencilBinding ds_binding = {&ssaa_ds_tex_, &ssaa_dsv_};

            const float offset_scale = (scale - 1) / 2;
            const D3D12_VIEWPORT viewports[]{{-offset_scale * ssaa_rt_tex_.Width(0), -offset_scale * ssaa_rt_tex_.Height(0),
                scale * ssaa_rt_tex_.Width(0), scale * ssaa_rt_tex_.Height(0), 0, 1}};
            const D3D12_RECT scissor_rcs[] = {{0, 0, static_cast<LONG>(ssaa_rt_tex_.Width(0)), static_cast<LONG>(ssaa_rt_tex_.Height(0))}};

            cmd_list.Render(
                render_pipeline_, vb_bindings, &ib_binding, num_indices, shader_bindings, rt_bindings, &ds_binding, viewports, scissor_rcs);
        }

        void Downsample(GpuCommandList& cmd_list, GpuTexture2D& target)
        {
            constexpr uint32_t BlockDim = 16;

            const GpuTexture2D* srv_texs[] = {&ssaa_rt_tex_};
            GpuTexture2D* uav_texs[] = {&target};
            const GpuCommandList::ShaderBinding shader_binding = {{}, srv_texs, uav_texs};
            cmd_list.Compute(downsample_pipeline_, DivUp(target.Width(0), BlockDim), DivUp(target.Height(0), BlockDim), 1, shader_binding);
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

            {
                const GpuTexture2D* srv_texs[] = {&ssaa_rt_tex_};
                GpuTexture2D* uav_texs[] = {&bb_tex_};
                const GpuCommandList::ShaderBinding shader_binding = {{}, srv_texs, uav_texs};
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
                const GpuTexture2D* srv_texs[] = {&mv_diffusion_tex};
                GpuTexture2D* uav_texs[] = {&bb_tex_};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srv_texs, uav_texs};
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
                const GpuTexture2D* srv_texs[] = {&mv_diffusion_tex, &bb_tex_};
                GpuTexture2D* uav_texs[] = {&ssaa_rt_tex_};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srv_texs, uav_texs};
                cmd_list.Compute(
                    blend_pipeline_, DivUp(ssaa_rt_tex_.Width(0), BlockDim), DivUp(ssaa_rt_tex_.Height(0), BlockDim), 1, shader_binding);
            }
        }

    private:
        GpuSystem& gpu_system_;
        PythonSystem& python_system_;

        GpuDescriptorBlock rtv_desc_block_;
        GpuDescriptorBlock dsv_desc_block_;

        GpuTexture2D init_view_tex_;
        GpuTexture2D multi_view_texs_[6];

        static constexpr uint32_t SsaaScale = 4;

        GpuTexture2D ssaa_rt_tex_;
        GpuRenderTargetView ssaa_rtv_;

        GpuTexture2D ssaa_ds_tex_;
        GpuDepthStencilView ssaa_dsv_;

        XMMATRIX proj_mtx_;

        struct RenderConstantBuffer
        {
            DirectX::XMFLOAT4X4 mvp;
        };
        ConstantBuffer<RenderConstantBuffer> render_cb_;
        GpuRenderPipeline render_pipeline_;

        GpuComputePipeline downsample_pipeline_;

        GpuComputePipeline calc_rendered_box_pipeline_;

        struct CalcDiffusionBoxConstantBuffer
        {
            XMUINT4 atlas_offset_view_size;
        };
        ConstantBuffer<CalcDiffusionBoxConstantBuffer> calc_diffusion_box_cb_;
        GpuComputePipeline calc_diffusion_box_pipeline_;

        struct BlendConstantBuffer
        {
            XMUINT4 atlas_offset_view_size;
            XMUINT4 rendered_diffusion_center;
            XMFLOAT4 diffusion_inv_size;
        };
        ConstantBuffer<BlendConstantBuffer> blend_cb_;
        GpuComputePipeline blend_pipeline_;
        GpuTexture2D bb_tex_;
    };

    MultiViewRenderer::MultiViewRenderer(GpuSystem& gpu_system, PythonSystem& python_system, uint32_t width, uint32_t height)
        : impl_(std::make_unique<Impl>(gpu_system, python_system, width, height))
    {
    }

    MultiViewRenderer::~MultiViewRenderer() noexcept = default;

    MultiViewRenderer::MultiViewRenderer(MultiViewRenderer&& other) noexcept = default;
    MultiViewRenderer& MultiViewRenderer::operator=(MultiViewRenderer&& other) noexcept = default;

    MultiViewRenderer::Result MultiViewRenderer::Render(const Mesh& mesh)
    {
        return impl_->Render(mesh);
    }
} // namespace AIHoloImager
