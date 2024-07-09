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
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON, L"ssaa_rt_tex_");
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

                const GpuRenderPipeline::ShaderInfo shaders[] = {
                    {GpuRenderPipeline::ShaderStage::Vertex, RenderVs_shader, 1, 0, 0},
                    {GpuRenderPipeline::ShaderStage::Pixel, RenderPs_shader, 0, 1, 0},
                };

                const DXGI_FORMAT rtv_formats[] = {ssaa_rt_tex_.Format()};

                GpuRenderPipeline::States states;
                states.cull_mode = GpuRenderPipeline::CullMode::ClockWise;
                states.depth_enable = true;
                states.rtv_formats = std::span(rtv_formats);
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

                render_pipeline_ = GpuRenderPipeline(gpu_system_, shaders, std::span(&point_sampler_desc, 1), states, input_elems);
            }

            downsample_shader_ = GpuComputeShader(gpu_system_, DownsampleCs_shader, 0, 1, 1, {});
        }

        ~Impl() noexcept
        {
            render_cb_ = ConstantBuffer<RenderConstantBuffer>();

            ssaa_dsv_.Reset();
            ssaa_ds_tex_.Reset();

            ssaa_rtv_.Reset();
            ssaa_rt_tex_.Reset();

            init_view_tex_.Reset();
            for (auto& tex : multi_view_texs_)
            {
                tex.Reset();
            }

            gpu_system_.DeallocDsvDescBlock(std::move(dsv_desc_block_));
            gpu_system_.DeallocRtvDescBlock(std::move(rtv_desc_block_));

            gpu_system_.WaitForGpu();
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

            GpuTexture2D albedo_tex;
            UploadGpuTexture(gpu_system_, mesh.AlbedoTexture(), albedo_tex);

            constexpr float CameraDist = 10;

            const uint32_t num_indices = static_cast<uint32_t>(mesh.Indices().size());

            {
                GpuCommandList cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                RenderToSsaa(cmd_list, vb, ib, num_indices, albedo_tex, 0, 45, CameraDist);
                Downsample(cmd_list, init_view_tex_);
                gpu_system_.Execute(std::move(cmd_list));
                gpu_system_.WaitForGpu();
            }

            Texture init_view_cpu_tex = ReadbackGpuTexture(gpu_system_, init_view_tex_);
            RemoveAlpha(init_view_cpu_tex);

            MultiViewDiffusion mv_diffusion(python_system_);
            Texture mv_diffusion_tex = mv_diffusion.Generate(init_view_cpu_tex);

            for (size_t i = 0; i < std::size(Azimuths); ++i)
            {
                {
                    GpuCommandList cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                    RenderToSsaa(cmd_list, vb, ib, num_indices, albedo_tex, Azimuths[i], Elevations[i], CameraDist, MvScale);
                    gpu_system_.Execute(std::move(cmd_list));
                    gpu_system_.WaitForGpu();
                }
                BlendWithDiffusion(mv_diffusion_tex, static_cast<uint32_t>(i));
                {
                    GpuCommandList cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                    Downsample(cmd_list, multi_view_texs_[i]);
                    gpu_system_.Execute(std::move(cmd_list));
                    gpu_system_.WaitForGpu();
                }
            }

            Result ret;
            for (uint32_t i = 0; i < 6; ++i)
            {
                ret.multi_view_images[i] = ReadbackGpuTexture(gpu_system_, multi_view_texs_[i]);
                RemoveAlpha(ret.multi_view_images[i]);
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

            const GeneralConstantBuffer* cb = &render_cb_;
            const GpuTexture2D* srv_tex = &albedo_tex;
            const GpuCommandList::ShaderBinding shader_bindings[] = {
                {GpuRenderPipeline::ShaderStage::Vertex, std::span(&cb, 1), {}, {}},
                {GpuRenderPipeline::ShaderStage::Pixel, {}, std::span(&srv_tex, 1), {}},
            };

            const GpuCommandList::RenderTargetBinding rt_bindings[] = {{&ssaa_rt_tex_, &ssaa_rtv_}};
            const GpuCommandList::DepthStencilBinding ds_binding = {&ssaa_ds_tex_, &ssaa_dsv_};

            const float offset_scale = (scale - 1) / 2;
            const D3D12_VIEWPORT viewport{-offset_scale * ssaa_rt_tex_.Width(0), -offset_scale * ssaa_rt_tex_.Height(0),
                scale * ssaa_rt_tex_.Width(0), scale * ssaa_rt_tex_.Height(0), 0, 1};
            const D3D12_RECT scissor_rc{0, 0, static_cast<LONG>(ssaa_rt_tex_.Width(0)), static_cast<LONG>(ssaa_rt_tex_.Height(0))};

            cmd_list.Render(vb_bindings, &ib_binding, num_indices, render_pipeline_, shader_bindings, rt_bindings, &ds_binding,
                std::span(&viewport, 1), std::span(&scissor_rc, 1));
        }

        void Downsample(GpuCommandList& cmd_list, GpuTexture2D& target)
        {
            constexpr uint32_t BlockDim = 16;

            const GpuTexture2D* srv_tex = &ssaa_rt_tex_;
            GpuTexture2D* uav_tex = &target;
            cmd_list.Compute(downsample_shader_, DivUp(target.Width(0), BlockDim), DivUp(target.Height(0), BlockDim), 1, {},
                std::span(&srv_tex, 1), std::span(&uav_tex, 1));
        }

        void BlendWithDiffusion(Texture& mv_diffusion_tex, uint32_t index)
        {
            // TODO #13: Port to GPU

            Texture rendered_read_back_tex = ReadbackGpuTexture(gpu_system_, ssaa_rt_tex_);

            XMUINT2 rendered_min(rendered_read_back_tex.Width(), rendered_read_back_tex.Height());
            XMUINT2 rendered_max(0, 0);
            uint8_t* rendered_data = rendered_read_back_tex.Data();
            const uint32_t rendered_width = rendered_read_back_tex.Width();
            const uint32_t rendered_channels = rendered_read_back_tex.NumChannels();
            for (uint32_t y = 0; y < rendered_read_back_tex.Height(); ++y)
            {
                for (uint32_t x = 0; x < rendered_read_back_tex.Width(); ++x)
                {
                    if (rendered_data[(y * rendered_width + x) * rendered_channels + 3] != 0)
                    {
                        rendered_min.x = std::min(rendered_min.x, x);
                        rendered_max.x = std::max(rendered_max.x, x);
                        rendered_min.y = std::min(rendered_min.y, y);
                        rendered_max.y = std::max(rendered_max.y, y);
                    }
                }
            }

            constexpr uint32_t AtlasWidth = 2;
            constexpr uint32_t AtlasHeight = 3;
            constexpr uint32_t ValidThreshold = 237;

            const uint32_t view_width = mv_diffusion_tex.Width() / AtlasWidth;
            const uint32_t view_height = mv_diffusion_tex.Height() / AtlasHeight;

            const uint32_t atlas_y = index / AtlasWidth;
            const uint32_t atlas_x = index - atlas_y * AtlasWidth;
            const uint32_t atlas_offset_x = atlas_x * view_width;
            const uint32_t atlas_offset_y = atlas_y * view_height;

            XMUINT2 diffusion_min(view_width, view_height);
            XMUINT2 diffusion_max(0, 0);
            const uint8_t* diffusion_data = mv_diffusion_tex.Data();
            const uint32_t diffusion_width = mv_diffusion_tex.Width();
            const uint32_t diffusion_channels = mv_diffusion_tex.NumChannels();
            for (uint32_t y = 0; y < view_height; ++y)
            {
                for (uint32_t x = 0; x < view_width; ++x)
                {
                    const uint32_t pixel_offset = ((atlas_offset_y + y) * diffusion_width + (atlas_offset_x + x)) * diffusion_channels;
                    for (uint32_t c = 0; c < 3; ++c)
                    {
                        if (diffusion_data[pixel_offset + c] < ValidThreshold)
                        {
                            diffusion_min.x = std::min(diffusion_min.x, x);
                            diffusion_max.x = std::max(diffusion_max.x, x);
                            diffusion_min.y = std::min(diffusion_min.y, y);
                            diffusion_max.y = std::max(diffusion_max.y, y);
                            break;
                        }
                    }
                }
            }

            const float scale_x = static_cast<float>(diffusion_max.x - diffusion_min.x) / (rendered_max.x - rendered_min.x);
            const float scale_y = static_cast<float>(diffusion_max.y - diffusion_min.y) / (rendered_max.y - rendered_min.y);
            const float scale = std::min(scale_x, scale_y);

            const auto is_empty = [](const uint8_t* rgb) -> bool {
                constexpr int empty_color_r = 0xFF;
                constexpr int empty_color_g = 0x7F;
                constexpr int empty_color_b = 0x27;
                return ((std::abs(static_cast<int>(rgb[0]) - empty_color_r) < 2) &&
                        (std::abs(static_cast<int>(rgb[1]) - empty_color_g) < 20) &&
                        (std::abs(static_cast<int>(rgb[2]) - empty_color_b) < 15));
            };

            const int rendered_center_x = rendered_read_back_tex.Width() / 2;
            const int rendered_center_y = rendered_read_back_tex.Height() / 2;
            const int diffusion_center_x = view_width / 2;
            const int diffusion_center_y = view_height / 2;
#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t y = 0; y < rendered_read_back_tex.Height(); ++y)
            {
                const uint32_t src_y = static_cast<uint32_t>(
                    std::clamp(static_cast<int>(std::round((static_cast<int>(y) - rendered_center_y) * scale)) + diffusion_center_y, 0,
                        static_cast<int>(view_height - 1)));
                for (uint32_t x = 0; x < rendered_read_back_tex.Width(); ++x)
                {
                    const uint32_t src_x = static_cast<uint32_t>(
                        std::clamp(static_cast<int>(std::round((static_cast<int>(x) - rendered_center_x) * scale)) + diffusion_center_x, 0,
                            static_cast<int>(view_width - 1)));

                    const uint32_t diffusion_offset = (atlas_offset_y + src_y) * diffusion_width + (atlas_offset_x + src_x);
                    const uint32_t rendered_offset = y * rendered_width + x;

                    bool rendered_valid = false;
                    if ((rendered_data[rendered_offset * rendered_channels + 3] != 0) &&
                        !is_empty(&rendered_data[rendered_offset * rendered_channels]))
                    {
                        rendered_valid = true;
                    }

                    bool diffusion_valid = false;
                    for (uint32_t c = 0; c < 3; ++c)
                    {
                        if (diffusion_data[diffusion_offset * diffusion_channels + c] < ValidThreshold)
                        {
                            diffusion_valid = true;
                            break;
                        }
                    }

                    if (!rendered_valid && diffusion_valid)
                    {
                        memcpy(&rendered_data[rendered_offset * rendered_channels], &diffusion_data[diffusion_offset * diffusion_channels],
                            diffusion_channels);
                        rendered_data[rendered_offset * rendered_channels + 3] = 0xFF;
                    }
                }
            }

            UploadGpuTexture(gpu_system_, rendered_read_back_tex, ssaa_rt_tex_);
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

        GpuComputeShader downsample_shader_;
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
