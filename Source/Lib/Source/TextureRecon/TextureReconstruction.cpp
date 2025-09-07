// Copyright (c) 2024-2025 Minmin Gong
//

#include "TextureReconstruction.hpp"

#include <format>
#include <iostream>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuSampler.hpp"

#include "CompiledShader/TextureRecon/FlattenPs.h"
#include "CompiledShader/TextureRecon/FlattenVs.h"
#include "CompiledShader/TextureRecon/GenShadowMapVs.h"
#include "CompiledShader/TextureRecon/ProjectTextureCs.h"
#include "CompiledShader/TextureRecon/ResolveTextureCs.h"

namespace AIHoloImager
{
    class TextureReconstruction::Impl
    {
    public:
        explicit Impl(AIHoloImagerInternal& aihi) : aihi_(aihi), gpu_system_(aihi.GpuSystemInstance())
        {
            const GpuVertexAttribs vertex_attribs(std::span<const GpuVertexAttrib>({
                {"POSITION", 0, GpuFormat::RGB32_Float},
                {"NORMAL", 0, GpuFormat::RGB32_Float},
                {"TEXCOORD", 0, GpuFormat::RG32_Float},
            }));

            {
                const ShaderInfo shaders[] = {
                    {FlattenVs_shader, 1, 0, 0},
                    {FlattenPs_shader, 0, 0, 0},
                };

                const GpuFormat rtv_formats[] = {PositionFmt, NormalFmt};

                GpuRenderPipeline::States states;
                states.cull_mode = GpuRenderPipeline::CullMode::None;
                states.conservative_raster = true;
                states.depth_enable = false;
                states.rtv_formats = rtv_formats;

                flatten_pipeline_ =
                    GpuRenderPipeline(gpu_system_, GpuRenderPipeline::PrimitiveTopology::TriangleList, shaders, vertex_attribs, {}, states);
            }
            {
                const ShaderInfo shaders[] = {
                    {GenShadowMapVs_shader, 1, 0, 0},
                };

                GpuRenderPipeline::States states;
                states.cull_mode = GpuRenderPipeline::CullMode::None;
                states.depth_enable = true;
                states.rtv_formats = {};
                states.dsv_format = DepthFmt;

                gen_shadow_map_pipeline_ =
                    GpuRenderPipeline(gpu_system_, GpuRenderPipeline::PrimitiveTopology::TriangleList, shaders, vertex_attribs, {}, states);
            }
            {
                const GpuStaticSampler samplers[] = {
                    GpuStaticSampler(
                        {GpuStaticSampler::Filter::Point, GpuStaticSampler::Filter::Point}, GpuStaticSampler::AddressMode::Clamp),
                    GpuStaticSampler(
                        {GpuStaticSampler::Filter::Linear, GpuStaticSampler::Filter::Linear}, GpuStaticSampler::AddressMode::Clamp),
                };

                const ShaderInfo shader = {ProjectTextureCs_shader, 1, 4, 1};
                project_texture_pipeline_ = GpuComputePipeline(gpu_system_, shader, std::span{samplers});
            }
            {
                const ShaderInfo shader = {ResolveTextureCs_shader, 1, 1, 1};
                resolve_texture_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
        }

        TextureReconstruction::Result Process(const Mesh& mesh, const glm::mat4x4& model_mtx, const Obb& world_obb,
            const StructureFromMotion::Result& sfm_input, uint32_t texture_size)
        {
            const uint32_t vertex_stride = mesh.MeshVertexDesc().Stride();

#ifdef AIHI_KEEP_INTERMEDIATES
            const auto output_dir = aihi_.TmpDir() / "Texture";
            std::filesystem::create_directories(output_dir);
#endif

            auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

            GpuBuffer mesh_vb(gpu_system_, static_cast<uint32_t>(mesh.VertexBuffer().size() * sizeof(float)), GpuHeap::Default,
                GpuResourceFlag::None, L"mesh_vb");
            cmd_list.Upload(mesh_vb, mesh.VertexBuffer().data(), mesh_vb.Size());

            GpuBuffer mesh_ib(gpu_system_, static_cast<uint32_t>(mesh.IndexBuffer().size() * sizeof(uint32_t)), GpuHeap::Default,
                GpuResourceFlag::None, L"mesh_ib");
            cmd_list.Upload(mesh_ib, mesh.IndexBuffer().data(), mesh_ib.Size());

            GpuTexture2D flatten_pos_tex;
            GpuTexture2D flatten_normal_tex;
            this->FlattenMesh(cmd_list, mesh_vb, vertex_stride, mesh_ib, model_mtx, texture_size, flatten_pos_tex, flatten_normal_tex);

#ifdef AIHI_KEEP_INTERMEDIATES
            {
                Texture pos_tex(flatten_normal_tex.Width(0), flatten_normal_tex.Height(0), ElementFormat::RGBA32_Float);
                const auto pos_rb_future = cmd_list.ReadBackAsync(flatten_pos_tex, 0, pos_tex.Data(), pos_tex.DataSize());

                Texture normal_tex(flatten_normal_tex.Width(0), flatten_normal_tex.Height(0), ElementFormat::RGBA8_UNorm);
                const auto normal_rb_future = cmd_list.ReadBackAsync(flatten_normal_tex, 0, normal_tex.Data(), normal_tex.DataSize());

                gpu_system_.ExecuteAndReset(cmd_list);

                pos_rb_future.wait();
                pos_tex.ConvertInPlace(ElementFormat::RGB32_Float);
                SaveTexture(pos_tex, output_dir / "FlattenPos.pfm");

                normal_rb_future.wait();
                SaveTexture(normal_tex, output_dir / "FlattenNormal.png");
            }
#endif

            TextureReconstruction::Result result;
            result.color_tex = this->GenTextureFromPhotos(cmd_list, mesh_vb, vertex_stride, mesh_ib, model_mtx, world_obb, flatten_pos_tex,
                flatten_normal_tex, sfm_input, texture_size);

#ifdef AIHI_KEEP_INTERMEDIATES
            {
                SaveMesh(mesh, output_dir / "MeshTextured.glb");

                Texture projective_tex(texture_size, texture_size, ElementFormat::RGBA8_UNorm);
                const auto rb_future = cmd_list.ReadBackAsync(result.color_tex, 0, projective_tex.Data(), projective_tex.DataSize());
                gpu_system_.ExecuteAndReset(cmd_list);

                rb_future.wait();
                SaveTexture(projective_tex, output_dir / "Projective.png");
            }
#endif

            gpu_system_.Execute(std::move(cmd_list));

            result.pos_tex = std::move(flatten_pos_tex);
            return result;
        }

    private:
        void FlattenMesh(GpuCommandList& cmd_list, const GpuBuffer& mesh_vb, uint32_t vertex_stride, const GpuBuffer& mesh_ib,
            const glm::mat4x4& model_mtx, uint32_t texture_size, GpuTexture2D& flatten_pos_tex, GpuTexture2D& flatten_normal_tex)
        {
            const uint32_t num_indices = static_cast<uint32_t>(mesh_ib.Size() / sizeof(uint32_t));

            flatten_pos_tex =
                GpuTexture2D(gpu_system_, texture_size, texture_size, 1, PositionFmt, GpuResourceFlag::RenderTarget, L"flatten_pos_tex");
            flatten_normal_tex =
                GpuTexture2D(gpu_system_, texture_size, texture_size, 1, NormalFmt, GpuResourceFlag::RenderTarget, L"flatten_normal_tex");

            GpuRenderTargetView pos_rtv(gpu_system_, flatten_pos_tex);
            GpuRenderTargetView normal_rtv(gpu_system_, flatten_normal_tex);

            GpuConstantBufferOfType<FlattenConstantBuffer> flatten_cb(gpu_system_, L"flatten_cb");
            flatten_cb->model_mtx = glm::transpose(model_mtx);
            flatten_cb->model_it_mtx = glm::inverse(model_mtx);
            flatten_cb.UploadStaging();

            const float clear_clr[] = {0, 0, 0, 0};
            cmd_list.Clear(pos_rtv, clear_clr);
            cmd_list.Clear(normal_rtv, clear_clr);

            const GpuCommandList::VertexBufferBinding vb_bindings[] = {{&mesh_vb, 0, vertex_stride}};
            const GpuCommandList::IndexBufferBinding ib_binding = {&mesh_ib, 0, GpuFormat::R32_Uint};

            const GpuConstantBuffer* cbs[] = {&flatten_cb};
            const GpuCommandList::ShaderBinding shader_bindings[] = {
                {cbs, {}, {}},
                {{}, {}, {}},
            };

            const GpuRenderTargetView* rtvs[] = {&pos_rtv, &normal_rtv};

            const GpuViewport viewports[] = {{0, 0, static_cast<float>(texture_size), static_cast<float>(texture_size)}};
            const GpuRect scissor_rcs[] = {{0, 0, static_cast<int32_t>(texture_size), static_cast<int32_t>(texture_size)}};

            cmd_list.Render(
                flatten_pipeline_, vb_bindings, &ib_binding, num_indices, shader_bindings, rtvs, nullptr, viewports, scissor_rcs);
        }

        GpuTexture2D GenTextureFromPhotos(GpuCommandList& cmd_list, const GpuBuffer& mesh_vb, uint32_t vertex_stride,
            const GpuBuffer& mesh_ib, const glm::mat4x4& model_mtx, const Obb& world_obb, const GpuTexture2D& flatten_pos_tex,
            const GpuTexture2D& flatten_normal_tex, const StructureFromMotion::Result& sfm_input, uint32_t texture_size)
        {
            const uint32_t num_indices = static_cast<uint32_t>(mesh_ib.Size() / sizeof(uint32_t));

            GpuTexture2D accum_color_tex(
                gpu_system_, texture_size, texture_size, 1, GpuFormat::RGBA8_UNorm, GpuResourceFlag::UnorderedAccess, L"accum_color_tex");
            GpuUnorderedAccessView accum_color_uav(gpu_system_, accum_color_tex);

            {
                const float black[] = {0, 0, 0, 0};
                cmd_list.Clear(accum_color_uav, black);
            }

            const GpuShaderResourceView flatten_pos_srv(gpu_system_, flatten_pos_tex, 0);
            const GpuShaderResourceView flatten_normal_srv(gpu_system_, flatten_normal_tex, 0);

            for (size_t i = 0; i < sfm_input.views.size(); ++i)
            {
                std::cout << std::format("Projecting images ({} / {})\r", i + 1, sfm_input.views.size());

                const auto& view = sfm_input.views[i];
                const auto& intrinsic = sfm_input.intrinsics[view.intrinsic_id];

                GpuTexture2D shadow_map_tex(gpu_system_, intrinsic.width, intrinsic.height, 1, GpuFormat::R32_Float,
                    GpuResourceFlag::DepthStencil, L"shadow_map_tex");
                const GpuShaderResourceView shadow_map_srv(gpu_system_, shadow_map_tex);
                GpuDepthStencilView shadow_map_dsv(gpu_system_, shadow_map_tex, DepthFmt);

                const GpuShaderResourceView photo_srv(gpu_system_, view.delighted_tex);

                const glm::mat4x4 view_mtx = CalcViewMatrix(view);
                const glm::vec2 near_far_plane = CalcNearFarPlane(view_mtx, world_obb);
                const glm::mat4x4 proj_mtx = CalcProjMatrix(intrinsic, near_far_plane.x, near_far_plane.y);
                const glm::mat4x4 mvp_mtx = proj_mtx * view_mtx * model_mtx;
                const glm::vec2 vp_offset = CalcViewportOffset(intrinsic);

                this->GenShadowMap(cmd_list, mesh_vb, vertex_stride, mesh_ib, num_indices, mvp_mtx, vp_offset, intrinsic, shadow_map_dsv);

                this->ProjectTexture(cmd_list, texture_size, view_mtx, proj_mtx, vp_offset, intrinsic, flatten_pos_srv, flatten_normal_srv,
                    photo_srv, view.delighted_offset, glm::uvec2(view.delighted_tex.Width(0), view.delighted_tex.Height(0)), shadow_map_srv,
                    accum_color_uav);

#ifdef AIHI_KEEP_INTERMEDIATES
                {
                    Texture color_tex(accum_color_tex.Width(0), accum_color_tex.Height(0), ElementFormat::RGBA8_UNorm);
                    const auto rb_future = cmd_list.ReadBackAsync(accum_color_tex, 0, color_tex.Data(), color_tex.DataSize());
                    gpu_system_.ExecuteAndReset(cmd_list);
                    rb_future.wait();
                    SaveTexture(color_tex, aihi_.TmpDir() / "Texture" / std::format("Projective_{}.png", i));
                }
#endif
            }
            std::cout << "\n";

            return this->ResolveTexture(cmd_list, texture_size, accum_color_tex);
        }

        void GenShadowMap(GpuCommandList& cmd_list, const GpuBuffer& vb, uint32_t vertex_stride, const GpuBuffer& ib, uint32_t num_indices,
            const glm::mat4x4& mvp_mtx, const glm::vec2& vp_offset, const StructureFromMotion::PinholeIntrinsic& intrinsic,
            GpuDepthStencilView& shadow_map_dsv)
        {
            GpuConstantBufferOfType<GenShadowMapConstantBuffer> gen_shadow_map_cb(gpu_system_, L"gen_shadow_map_cb");
            gen_shadow_map_cb->mvp = glm::transpose(mvp_mtx);
            gen_shadow_map_cb.UploadStaging();

            cmd_list.ClearDepth(shadow_map_dsv, 1);

            const GpuCommandList::VertexBufferBinding vb_bindings[] = {{&vb, 0, vertex_stride}};
            const GpuCommandList::IndexBufferBinding ib_binding = {&ib, 0, GpuFormat::R32_Uint};

            const GpuConstantBuffer* cbs[] = {&gen_shadow_map_cb};
            const GpuCommandList::ShaderBinding shader_bindings[] = {
                {cbs, {}, {}},
                {{}, {}, {}},
            };

            const GpuViewport viewport = {
                vp_offset.x, vp_offset.y, static_cast<float>(intrinsic.width), static_cast<float>(intrinsic.height)};

            cmd_list.Render(gen_shadow_map_pipeline_, vb_bindings, &ib_binding, num_indices, shader_bindings, {}, &shadow_map_dsv,
                std::span(&viewport, 1), {});
        }

        void ProjectTexture(GpuCommandList& cmd_list, uint32_t texture_size, const glm::mat4x4& view_mtx, const glm::mat4x4& proj_mtx,
            const glm::vec2& vp_offset, const StructureFromMotion::PinholeIntrinsic& intrinsic,
            const GpuShaderResourceView& flatten_pos_srv, const GpuShaderResourceView& flatten_normal_srv,
            const GpuShaderResourceView& projective_map_srv, const glm::uvec2& delighted_offset, const glm::uvec2& delighted_size,
            const GpuShaderResourceView& shadow_map_srv, GpuUnorderedAccessView& accum_color_uav)
        {
            constexpr uint32_t BlockDim = 16;

            GpuConstantBufferOfType<ProjectTextureConstantBuffer> project_tex_cb(gpu_system_, L"project_tex_cb");
            project_tex_cb->camera_view_proj = glm::transpose(proj_mtx * view_mtx);
            project_tex_cb->camera_view = glm::transpose(view_mtx);
            project_tex_cb->camera_view_it = glm::inverse(view_mtx);
            project_tex_cb->vp_offset = glm::vec2(vp_offset.x / intrinsic.width, vp_offset.y / intrinsic.height);
            project_tex_cb->delighted_offset = glm::vec2(
                static_cast<float>(delighted_offset.x) / intrinsic.width, static_cast<float>(delighted_offset.y) / intrinsic.height);
            project_tex_cb->delighted_scale =
                glm::vec2(static_cast<float>(intrinsic.width) / delighted_size.x, static_cast<float>(intrinsic.height) / delighted_size.y);
            project_tex_cb->texture_size = texture_size;
            project_tex_cb.UploadStaging();

            const GpuConstantBuffer* cbs[] = {&project_tex_cb};
            const GpuShaderResourceView* srvs[] = {&flatten_pos_srv, &flatten_normal_srv, &projective_map_srv, &shadow_map_srv};
            GpuUnorderedAccessView* uavs[] = {&accum_color_uav};

            const GpuCommandList::ShaderBinding cs_shader_binding = {cbs, srvs, uavs};
            cmd_list.Compute(project_texture_pipeline_, DivUp(texture_size, BlockDim), DivUp(texture_size, BlockDim), 1, cs_shader_binding);
        }

        GpuTexture2D ResolveTexture(GpuCommandList& cmd_list, uint32_t texture_size, const GpuTexture2D& accum_color_tex)
        {
            constexpr uint32_t BlockDim = 16;

            GpuTexture2D color_tex(gpu_system_, texture_size, texture_size, 1, ColorFmt, GpuResourceFlag::UnorderedAccess, L"color_tex");
            GpuUnorderedAccessView color_uav(gpu_system_, color_tex);

            const GpuShaderResourceView accum_color_srv(gpu_system_, accum_color_tex);

            GpuConstantBufferOfType<ResolveTextureConstantBuffer> resolve_texture_cb(gpu_system_, L"resolve_texture_cb");
            resolve_texture_cb->texture_size = texture_size;
            resolve_texture_cb.UploadStaging();

            const GpuConstantBuffer* cbs[] = {&resolve_texture_cb};
            const GpuShaderResourceView* srvs[] = {&accum_color_srv};
            GpuUnorderedAccessView* uavs[] = {&color_uav};

            const GpuCommandList::ShaderBinding cs_shader_binding = {cbs, srvs, uavs};
            cmd_list.Compute(resolve_texture_pipeline_, DivUp(texture_size, BlockDim), DivUp(texture_size, BlockDim), 1, cs_shader_binding);

            return color_tex;
        }

    private:
        AIHoloImagerInternal& aihi_;
        GpuSystem& gpu_system_;

        struct FlattenConstantBuffer
        {
            glm::mat4x4 model_mtx;
            glm::mat4x4 model_it_mtx;
        };
        GpuRenderPipeline flatten_pipeline_;

        struct GenShadowMapConstantBuffer
        {
            glm::mat4x4 mvp;
        };
        GpuRenderPipeline gen_shadow_map_pipeline_;

        struct ProjectTextureConstantBuffer
        {
            glm::mat4x4 camera_view_proj;
            glm::mat4x4 camera_view;
            glm::mat4x4 camera_view_it;
            glm::vec2 vp_offset;
            glm::vec2 delighted_offset;
            glm::vec2 delighted_scale;
            uint32_t texture_size;
            uint32_t padding;
        };
        GpuComputePipeline project_texture_pipeline_;

        struct ResolveTextureConstantBuffer
        {
            uint32_t texture_size;
            uint32_t padding[3];
        };
        GpuComputePipeline resolve_texture_pipeline_;

        static constexpr GpuFormat ColorFmt = GpuFormat::RGBA8_UNorm;
        static constexpr GpuFormat PositionFmt = GpuFormat::RGBA32_Float;
        static constexpr GpuFormat NormalFmt = GpuFormat::RGBA8_UNorm;
        static constexpr GpuFormat DepthFmt = GpuFormat::D32_Float;
    };

    TextureReconstruction::TextureReconstruction(AIHoloImagerInternal& aihi) : impl_(std::make_unique<Impl>(aihi))
    {
    }

    TextureReconstruction::~TextureReconstruction() noexcept = default;

    TextureReconstruction::TextureReconstruction(TextureReconstruction&& other) noexcept = default;
    TextureReconstruction& TextureReconstruction::operator=(TextureReconstruction&& other) noexcept = default;

    TextureReconstruction::Result TextureReconstruction::Process(const Mesh& mesh, const glm::mat4x4& model_mtx, const Obb& world_obb,
        const StructureFromMotion::Result& sfm_input, uint32_t texture_size)
    {
        return impl_->Process(mesh, model_mtx, world_obb, sfm_input, texture_size);
    }
} // namespace AIHoloImager
