// Copyright (c) 2024 Minmin Gong
//

#include "TextureReconstruction.hpp"

#include <iostream>
#ifdef AIHI_KEEP_INTERMEDIATES
    #include <format>
#endif

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"

#include "CompiledShader/FlattenPs.h"
#include "CompiledShader/FlattenVs.h"
#include "CompiledShader/GenShadowMapVs.h"
#include "CompiledShader/ProjectTextureCs.h"
#include "CompiledShader/ResolveTextureCs.h"

using namespace DirectX;

namespace AIHoloImager
{
    class TextureReconstruction::Impl
    {
    public:
        Impl(const std::filesystem::path& exe_dir, GpuSystem& gpu_system) : exe_dir_(exe_dir), gpu_system_(gpu_system)
        {
            const D3D12_INPUT_ELEMENT_DESC input_elems[] = {
                {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            };

            {
                flatten_cb_ = ConstantBuffer<FlattenConstantBuffer>(gpu_system_, 1, L"flatten_cb_");

                const ShaderInfo shaders[] = {
                    {FlattenVs_shader, 1, 0, 0},
                    {FlattenPs_shader, 0, 0, 0},
                };

                const DXGI_FORMAT rtv_formats[] = {PositionFmt, NormalFmt};

                GpuRenderPipeline::States states;
                states.cull_mode = GpuRenderPipeline::CullMode::None;
                states.conservative_raster = true;
                states.depth_enable = false;
                states.rtv_formats = rtv_formats;
                states.dsv_format = DXGI_FORMAT_UNKNOWN;

                flatten_pipeline_ = GpuRenderPipeline(gpu_system_, shaders, input_elems, {}, states);
            }
            {
                gen_shadow_map_cb_ = ConstantBuffer<GenShadowMapConstantBuffer>(gpu_system_, 1, L"gen_shadow_map_cb_");

                const ShaderInfo shaders[] = {
                    {GenShadowMapVs_shader, 1, 0, 0},
                    {{}, 0, 0, 0},
                };

                GpuRenderPipeline::States states;
                states.cull_mode = GpuRenderPipeline::CullMode::None;
                states.depth_enable = true;
                states.rtv_formats = {};
                states.dsv_format = DepthFmt;

                gen_shadow_map_pipeline_ = GpuRenderPipeline(gpu_system_, shaders, input_elems, {}, states);
            }
            {
                project_tex_cb_ = ConstantBuffer<ProjectTextureConstantBuffer>(gpu_system_, 1, L"project_tex_cb_");

                D3D12_STATIC_SAMPLER_DESC sampler_desc[2]{};

                auto& point_sampler_desc = sampler_desc[0];
                point_sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
                point_sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                point_sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                point_sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                point_sampler_desc.MaxAnisotropy = 16;
                point_sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
                point_sampler_desc.MinLOD = 0.0f;
                point_sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
                point_sampler_desc.ShaderRegister = 0;

                auto& bilinear_sampler_desc = sampler_desc[1];
                bilinear_sampler_desc.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
                bilinear_sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                bilinear_sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                bilinear_sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                bilinear_sampler_desc.MaxAnisotropy = 16;
                bilinear_sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
                bilinear_sampler_desc.MinLOD = 0.0f;
                bilinear_sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
                bilinear_sampler_desc.ShaderRegister = 1;

                const ShaderInfo shader = {ProjectTextureCs_shader, 1, 4, 1};
                project_texture_pipeline_ = GpuComputePipeline(gpu_system_, shader, std::span{sampler_desc});
            }
            {
                resolve_texture_cb_ = ConstantBuffer<ResolveTextureConstantBuffer>(gpu_system_, 1, L"resolve_texture_cb_");

                const ShaderInfo shader = {ResolveTextureCs_shader, 1, 1, 1};
                resolve_texture_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
        }

        TextureReconstruction::Result Process(const Mesh& mesh, const XMMATRIX& model_mtx, const DirectX::BoundingOrientedBox& world_obb,
            const StructureFromMotion::Result& sfm_input, uint32_t texture_size, const std::filesystem::path& tmp_dir)
        {
            assert(mesh.MeshVertexDesc().Stride() == sizeof(VertexFormat));

#ifdef AIHI_KEEP_INTERMEDIATES
            const auto output_dir = tmp_dir / "Texture";
            std::filesystem::create_directories(output_dir);
#endif

            GpuBuffer mesh_vb(gpu_system_, static_cast<uint32_t>(mesh.VertexBuffer().size() * sizeof(float)), D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, L"mesh_vb");
            memcpy(mesh_vb.Map(), mesh.VertexBuffer().data(), mesh_vb.Size());
            mesh_vb.Unmap(D3D12_RANGE{0, mesh_vb.Size()});

            GpuBuffer mesh_ib(gpu_system_, static_cast<uint32_t>(mesh.IndexBuffer().size() * sizeof(uint32_t)), D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, L"mesh_ib");
            memcpy(mesh_ib.Map(), mesh.IndexBuffer().data(), mesh_ib.Size());
            mesh_ib.Unmap(D3D12_RANGE{0, mesh_ib.Size()});

            const XMMATRIX handedness = XMMatrixScaling(1, 1, -1);
            const XMMATRIX model_mtx_lh = model_mtx * handedness;

            GpuTexture2D flatten_pos_tex;
            GpuTexture2D flatten_normal_tex;
            this->FlattenMesh(mesh_vb, mesh_ib, model_mtx_lh, texture_size, flatten_pos_tex, flatten_normal_tex);

#ifdef AIHI_KEEP_INTERMEDIATES
            {
                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                Texture normal_tex(flatten_normal_tex.Width(0), flatten_normal_tex.Height(0), 4);
                flatten_normal_tex.Readback(gpu_system_, cmd_list, 0, normal_tex.Data());
                gpu_system_.Execute(std::move(cmd_list));

                SaveTexture(normal_tex, output_dir / "FlattenNormal.png");
            }
#endif

            TextureReconstruction::Result result;
            result.color_tex = this->GenTextureFromPhotos(
                mesh_vb, mesh_ib, model_mtx_lh, world_obb, flatten_pos_tex, flatten_normal_tex, sfm_input, texture_size, tmp_dir);

#ifdef AIHI_KEEP_INTERMEDIATES
            {
                SaveMesh(mesh, output_dir / "MeshTextured.glb");

                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                Texture projective_tex(texture_size, texture_size, 4);
                result.color_tex.Readback(gpu_system_, cmd_list, 0, projective_tex.Data());
                gpu_system_.Execute(std::move(cmd_list));

                SaveTexture(projective_tex, output_dir / "Projective.png");
            }
#endif

            result.pos_tex = std::move(flatten_pos_tex);
            XMStoreFloat4x4(&result.inv_model, XMMatrixInverse(nullptr, model_mtx_lh));

            return result;
        }

    private:
        void FlattenMesh(const GpuBuffer& mesh_vb, const GpuBuffer& mesh_ib, const XMMATRIX& model_mtx, uint32_t texture_size,
            GpuTexture2D& flatten_pos_tex, GpuTexture2D& flatten_normal_tex)
        {
            const uint32_t num_indices = static_cast<uint32_t>(mesh_ib.Size() / sizeof(uint32_t));

            flatten_pos_tex = GpuTexture2D(gpu_system_, texture_size, texture_size, 1, PositionFmt, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COMMON, L"flatten_pos_tex");
            flatten_normal_tex = GpuTexture2D(gpu_system_, texture_size, texture_size, 1, NormalFmt,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON, L"flatten_normal_tex");

            GpuRenderTargetView pos_rtv(gpu_system_, flatten_pos_tex);
            GpuRenderTargetView normal_rtv(gpu_system_, flatten_normal_tex);

            GpuCommandList cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

            XMStoreFloat4x4(&flatten_cb_->model_mtx, XMMatrixTranspose(model_mtx));
            XMStoreFloat4x4(&flatten_cb_->model_it_mtx, XMMatrixInverse(nullptr, model_mtx));
            flatten_cb_.UploadToGpu();

            const float clear_clr[] = {0, 0, 0, 0};
            cmd_list.Clear(pos_rtv, clear_clr);
            cmd_list.Clear(normal_rtv, clear_clr);

            const GpuCommandList::VertexBufferBinding vb_bindings[] = {{&mesh_vb, 0, sizeof(VertexFormat)}};
            const GpuCommandList::IndexBufferBinding ib_binding = {&mesh_ib, 0, DXGI_FORMAT_R32_UINT};

            const GeneralConstantBuffer* cbs[] = {&flatten_cb_};
            const GpuCommandList::ShaderBinding shader_bindings[] = {
                {cbs, {}, {}},
                {{}, {}, {}},
            };

            const GpuRenderTargetView* rtvs[] = {&pos_rtv, &normal_rtv};

            const D3D12_VIEWPORT viewports[] = {{0, 0, static_cast<float>(texture_size), static_cast<float>(texture_size), 0, 1}};
            const D3D12_RECT scissor_rcs[] = {{0, 0, static_cast<LONG>(texture_size), static_cast<LONG>(texture_size)}};

            cmd_list.Render(
                flatten_pipeline_, vb_bindings, &ib_binding, num_indices, shader_bindings, rtvs, nullptr, viewports, scissor_rcs);

            flatten_pos_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);
            flatten_normal_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);

            gpu_system_.Execute(std::move(cmd_list));
            gpu_system_.WaitForGpu();
        }

        GpuTexture2D GenTextureFromPhotos(const GpuBuffer& mesh_vb, const GpuBuffer& mesh_ib, const XMMATRIX& model_mtx,
            const BoundingOrientedBox& world_obb, const GpuTexture2D& flatten_pos_tex, const GpuTexture2D& flatten_normal_tex,
            const StructureFromMotion::Result& sfm_input, uint32_t texture_size, [[maybe_unused]] const std::filesystem::path& tmp_dir)
        {
            const uint32_t num_indices = static_cast<uint32_t>(mesh_ib.Size() / sizeof(uint32_t));

            GpuTexture2D accum_color_tex(gpu_system_, texture_size, texture_size, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, L"accum_color_tex");
            GpuUnorderedAccessView accum_color_uav(gpu_system_, accum_color_tex);

            {
                GpuCommandList cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                const float black[] = {0, 0, 0, 0};
                cmd_list.Clear(accum_color_uav, black);
                gpu_system_.Execute(std::move(cmd_list));
            }

            GpuShaderResourceView flatten_pos_srv(gpu_system_, flatten_pos_tex, 0);
            GpuShaderResourceView flatten_normal_srv(gpu_system_, flatten_normal_tex, 0);

            GpuTexture2D shadow_map_tex;
            GpuShaderResourceView shadow_map_srv;
            GpuDepthStencilView shadow_map_dsv;

            GpuTexture2D photo_tex;
            GpuShaderResourceView photo_srv;

            for (size_t i = 0; i < sfm_input.views.size(); ++i)
            {
                std::cout << "Projecting images (" << (i + 1) << " / " << sfm_input.views.size() << ")\r";

                const auto& view = sfm_input.views[i];
                const auto& intrinsic = sfm_input.intrinsics[view.intrinsic_id];

                GpuCommandList cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

                if ((intrinsic.width != shadow_map_tex.Width(0)) || (intrinsic.height != shadow_map_tex.Height(0)))
                {
                    shadow_map_tex = GpuTexture2D(gpu_system_, intrinsic.width, intrinsic.height, 1, DXGI_FORMAT_R32_FLOAT,
                        D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_COMMON, L"shadow_map_tex");
                    shadow_map_srv = GpuShaderResourceView(gpu_system_, shadow_map_tex);
                    shadow_map_dsv = GpuDepthStencilView(gpu_system_, shadow_map_tex, DepthFmt);
                }

                if ((view.image_mask.Width() != photo_tex.Width(0)) || (view.image_mask.Height() != photo_tex.Height(0)))
                {
                    photo_tex = GpuTexture2D(gpu_system_, view.image_mask.Width(), view.image_mask.Height(), 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, L"photo_tex");
                    photo_srv = GpuShaderResourceView(gpu_system_, photo_tex);
                }
                photo_tex.Upload(gpu_system_, cmd_list, 0, view.image_mask.Data());

                const XMVECTOR camera_pos = XMVectorSet(
                    static_cast<float>(view.center.x()), static_cast<float>(view.center.y()), -static_cast<float>(view.center.z()), 1);
                const XMVECTOR camera_up_vec = XMVectorSet(-static_cast<float>(view.rotation(1, 0)),
                    -static_cast<float>(view.rotation(1, 1)), static_cast<float>(view.rotation(1, 2)), 0);
                const XMVECTOR camera_forward_vec = XMVectorSet(static_cast<float>(view.rotation(2, 0)),
                    static_cast<float>(view.rotation(2, 1)), -static_cast<float>(view.rotation(2, 2)), 0);

                const XMMATRIX view_mtx = XMMatrixLookAtLH(camera_pos, camera_pos + camera_forward_vec, camera_up_vec);

                XMFLOAT3 corners[BoundingOrientedBox::CORNER_COUNT];
                world_obb.GetCorners(corners);

                const XMVECTOR z_col = XMVectorSet(
                    XMVectorGetZ(view_mtx.r[0]), XMVectorGetZ(view_mtx.r[1]), XMVectorGetZ(view_mtx.r[2]), XMVectorGetZ(view_mtx.r[3]));

                float min_z_es = 1e10f;
                float max_z_es = -1e10f;
                for (const auto& corner : corners)
                {
                    XMVECTOR pos = XMVectorSet(corner.x, corner.y, -corner.z, 1);
                    const float z = XMVectorGetZ(XMVector4Dot(pos, z_col));
                    min_z_es = std::min(min_z_es, z);
                    max_z_es = std::max(max_z_es, z);
                }

                const float center_es_z = (max_z_es + min_z_es) / 2;
                const float extent_es_z = (max_z_es - min_z_es) / 2 * 1.05f;

                const float near_plane = center_es_z - extent_es_z;
                const float far_plane = center_es_z + extent_es_z;

                const double fy = intrinsic.k(1, 1);
                const float fov = static_cast<float>(2 * std::atan(intrinsic.height / (2 * fy)));
                const XMMATRIX proj_mtx =
                    XMMatrixPerspectiveFovLH(fov, static_cast<float>(intrinsic.width) / intrinsic.height, near_plane, far_plane);

                XMStoreFloat4x4(&gen_shadow_map_cb_->mvp, XMMatrixTranspose(model_mtx * view_mtx * proj_mtx));
                gen_shadow_map_cb_.UploadToGpu();

                const XMFLOAT2 offset = {
                    static_cast<float>(intrinsic.k(0, 2)) - intrinsic.width / 2,
                    static_cast<float>(intrinsic.k(1, 2)) - intrinsic.height / 2,
                };

                shadow_map_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_DEPTH_WRITE);

                this->GenShadowMap(cmd_list, mesh_vb, mesh_ib, num_indices, offset, intrinsic, shadow_map_dsv);

                shadow_map_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);
                accum_color_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

                this->ProjectTexture(cmd_list, texture_size, view_mtx, proj_mtx, offset, intrinsic, flatten_pos_srv, flatten_normal_srv,
                    photo_srv, shadow_map_srv, accum_color_uav);

                accum_color_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);

#ifdef AIHI_KEEP_INTERMEDIATES
                {
                    Texture color_tex(accum_color_tex.Width(0), accum_color_tex.Height(0), 4);
                    accum_color_tex.Readback(gpu_system_, cmd_list, 0, color_tex.Data());
                    SaveTexture(color_tex, tmp_dir / "Texture" / std::format("Projective_{}.png", i));
                }
#endif

                gpu_system_.Execute(std::move(cmd_list));
                gpu_system_.WaitForGpu();
            }
            std::cout << "\n";

            return this->ResolveTexture(texture_size, accum_color_tex);
        }

        void GenShadowMap(GpuCommandList& cmd_list, const GpuBuffer& vb, const GpuBuffer& ib, uint32_t num_indices, const XMFLOAT2& offset,
            const StructureFromMotion::PinholeIntrinsic& intrinsic, GpuDepthStencilView& shadow_map_dsv)
        {
            cmd_list.ClearDepth(shadow_map_dsv, 1);

            const GpuCommandList::VertexBufferBinding vb_bindings[] = {{&vb, 0, sizeof(VertexFormat)}};
            const GpuCommandList::IndexBufferBinding ib_binding = {&ib, 0, DXGI_FORMAT_R32_UINT};

            const GeneralConstantBuffer* cbs[] = {&gen_shadow_map_cb_};
            const GpuCommandList::ShaderBinding shader_bindings[] = {
                {cbs, {}, {}},
                {{}, {}, {}},
            };

            const D3D12_VIEWPORT viewports[] = {
                {offset.x, offset.y, static_cast<float>(intrinsic.width), static_cast<float>(intrinsic.height), 0, 1}};
            const D3D12_RECT scissor_rcs[] = {{0, 0, static_cast<LONG>(intrinsic.width), static_cast<LONG>(intrinsic.height)}};

            cmd_list.Render(gen_shadow_map_pipeline_, vb_bindings, &ib_binding, num_indices, shader_bindings, {}, &shadow_map_dsv,
                viewports, scissor_rcs);
        }

        void ProjectTexture(GpuCommandList& cmd_list, uint32_t texture_size, const XMMATRIX& view_mtx, const XMMATRIX& proj_mtx,
            const XMFLOAT2& offset, const StructureFromMotion::PinholeIntrinsic& intrinsic, const GpuShaderResourceView& flatten_pos_srv,
            const GpuShaderResourceView& flatten_normal_srv, const GpuShaderResourceView& projective_map_srv,
            const GpuShaderResourceView& shadow_map_srv, GpuUnorderedAccessView& accum_color_uav)
        {
            constexpr uint32_t BlockDim = 16;

            XMStoreFloat4x4(&project_tex_cb_->camera_view_proj, XMMatrixTranspose(view_mtx * proj_mtx));
            XMStoreFloat4x4(&project_tex_cb_->camera_view, XMMatrixTranspose(view_mtx));
            XMStoreFloat4x4(&project_tex_cb_->camera_view_it, XMMatrixInverse(nullptr, view_mtx));
            project_tex_cb_->offset = XMFLOAT2(offset.x / intrinsic.width, offset.y / intrinsic.height);
            project_tex_cb_->texture_size = texture_size;
            project_tex_cb_.UploadToGpu();

            const GeneralConstantBuffer* cbs[] = {&project_tex_cb_};
            const GpuShaderResourceView* srvs[] = {&flatten_pos_srv, &flatten_normal_srv, &projective_map_srv, &shadow_map_srv};
            GpuUnorderedAccessView* uavs[] = {&accum_color_uav};

            const GpuCommandList::ShaderBinding cs_shader_binding = {cbs, srvs, uavs};
            cmd_list.Compute(project_texture_pipeline_, DivUp(texture_size, BlockDim), DivUp(texture_size, BlockDim), 1, cs_shader_binding);
        }

        GpuTexture2D ResolveTexture(uint32_t texture_size, const GpuTexture2D& accum_color_tex)
        {
            constexpr uint32_t BlockDim = 16;

            GpuCommandList cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

            GpuTexture2D color_tex(gpu_system_, texture_size, texture_size, 1, ColorFmt, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON, L"color_tex");
            GpuUnorderedAccessView color_uav(gpu_system_, color_tex);

            GpuShaderResourceView accum_color_srv(gpu_system_, accum_color_tex);

            resolve_texture_cb_->texture_size = texture_size;
            resolve_texture_cb_.UploadToGpu();

            const GeneralConstantBuffer* cbs[] = {&resolve_texture_cb_};
            const GpuShaderResourceView* srvs[] = {&accum_color_srv};
            GpuUnorderedAccessView* uavs[] = {&color_uav};

            const GpuCommandList::ShaderBinding cs_shader_binding = {cbs, srvs, uavs};
            cmd_list.Compute(resolve_texture_pipeline_, DivUp(texture_size, BlockDim), DivUp(texture_size, BlockDim), 1, cs_shader_binding);

            gpu_system_.Execute(std::move(cmd_list));

            return color_tex;
        }

    private:
        const std::filesystem::path exe_dir_;

        GpuSystem& gpu_system_;

        struct VertexFormat
        {
            XMFLOAT3 pos;
            XMFLOAT3 normal;
            XMFLOAT2 texcoord;
        };

        struct FlattenConstantBuffer
        {
            XMFLOAT4X4 model_mtx;
            XMFLOAT4X4 model_it_mtx;
        };
        ConstantBuffer<FlattenConstantBuffer> flatten_cb_;
        GpuRenderPipeline flatten_pipeline_;

        struct GenShadowMapConstantBuffer
        {
            XMFLOAT4X4 mvp;
        };
        ConstantBuffer<GenShadowMapConstantBuffer> gen_shadow_map_cb_;
        GpuRenderPipeline gen_shadow_map_pipeline_;

        struct ProjectTextureConstantBuffer
        {
            XMFLOAT4X4 camera_view_proj;
            XMFLOAT4X4 camera_view;
            XMFLOAT4X4 camera_view_it;
            XMFLOAT2 offset;
            uint32_t texture_size;
            uint32_t padding;
        };
        ConstantBuffer<ProjectTextureConstantBuffer> project_tex_cb_;
        GpuComputePipeline project_texture_pipeline_;

        struct ResolveTextureConstantBuffer
        {
            uint32_t texture_size;
            uint32_t padding[3];
        };
        ConstantBuffer<ResolveTextureConstantBuffer> resolve_texture_cb_;
        GpuComputePipeline resolve_texture_pipeline_;

        static constexpr DXGI_FORMAT ColorFmt = DXGI_FORMAT_R8G8B8A8_UNORM;
        static constexpr DXGI_FORMAT PositionFmt = DXGI_FORMAT_R32G32B32A32_FLOAT;
        static constexpr DXGI_FORMAT NormalFmt = DXGI_FORMAT_R8G8B8A8_UNORM;
        static constexpr DXGI_FORMAT DepthFmt = DXGI_FORMAT_D32_FLOAT;
    };

    TextureReconstruction::TextureReconstruction(const std::filesystem::path& exe_dir, GpuSystem& gpu_system)
        : impl_(std::make_unique<Impl>(exe_dir, gpu_system))
    {
    }

    TextureReconstruction::~TextureReconstruction() noexcept = default;

    TextureReconstruction::TextureReconstruction(TextureReconstruction&& other) noexcept = default;
    TextureReconstruction& TextureReconstruction::operator=(TextureReconstruction&& other) noexcept = default;

    TextureReconstruction::Result TextureReconstruction::Process(const Mesh& mesh, const XMMATRIX& model_mtx,
        const BoundingOrientedBox& world_obb, const StructureFromMotion::Result& sfm_input, uint32_t texture_size,
        const std::filesystem::path& tmp_dir)
    {
        return impl_->Process(mesh, model_mtx, world_obb, sfm_input, texture_size, tmp_dir);
    }
} // namespace AIHoloImager
