// Copyright (c) 2025 Minmin Gong
//

#include "InvisibleFacesRemover.hpp"

#include <algorithm>
#include <format>
#include <numbers>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "Gpu/GpuBufferHelper.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuShader.hpp"
#include "Gpu/GpuTexture.hpp"

#include "CompiledShader/MeshGen/InvisibleFacesRemover/AccumFacesCs.h"
#include "CompiledShader/MeshGen/InvisibleFacesRemover/FaceIdPs.h"
#include "CompiledShader/MeshGen/InvisibleFacesRemover/FaceIdVs.h"
#include "CompiledShader/MeshGen/InvisibleFacesRemover/FilterFacesCs.h"
#include "CompiledShader/MeshGen/InvisibleFacesRemover/MarkFacesCs.h"

using namespace AIHoloImager;

namespace
{
    float RadicalInverse(uint32_t n, uint32_t base)
    {
        float inv_base = 1.0f / base;
        float f = inv_base;
        float result = 0;
        while (n > 0)
        {
            result += f * (n % base);
            n /= base;
            f *= inv_base;
        }
        return result;
    }

    float HaltonSequence(uint32_t index)
    {
        return RadicalInverse(index, 2);
    }

    glm::vec2 HammersleySequence(uint32_t index, uint32_t num_samples)
    {
        return glm::vec2(static_cast<float>(index) / num_samples, HaltonSequence(index));
    }

    glm::vec2 SphereHammersleySequence(uint32_t index, uint32_t num_samples)
    {
        const glm::vec2 uv = HammersleySequence(index, num_samples);

        const float theta = std::acos(1 - 2 * uv.x) - std::numbers::pi_v<float> / 2;
        const float phi = uv.y * 2 * std::numbers::pi_v<float>;
        return glm::vec2(phi, theta);
    }

    glm::vec3 SphericalCameraPose(float azimuth, float elevation, float radius)
    {
        const float sin_azimuth = std::sin(azimuth);
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
    class InvisibleFacesRemover::Impl
    {
    public:
        explicit Impl(AIHoloImagerInternal& aihi)
            : gpu_system_(aihi.GpuSystemInstance()), proj_mtx_(glm::perspectiveRH_ZO(Fov, 1.0f, 1.0f, 3.0f))
        {
            face_id_tex_ = GpuTexture2D(gpu_system_, RtSize, RtSize, 1, GpuFormat::R32_Uint,
                GpuResourceFlag::RenderTarget | GpuResourceFlag::UnorderedAccess, L"face_id_tex_");
            face_id_rtv_ = GpuRenderTargetView(gpu_system_, face_id_tex_);
            face_id_srv_ = GpuShaderResourceView(gpu_system_, face_id_tex_);

            ds_tex_ = GpuTexture2D(gpu_system_, RtSize, RtSize, 1, GpuFormat::D32_Float, GpuResourceFlag::DepthStencil, L"ds_tex_");
            dsv_ = GpuDepthStencilView(gpu_system_, ds_tex_);

            filtered_counter_buff_ =
                GpuBuffer(gpu_system_, sizeof(uint32_t), GpuHeap::Default, GpuResourceFlag::UnorderedAccess, L"filtered_counter_buff_");
            filtered_counter_uav_ = GpuUnorderedAccessView(gpu_system_, filtered_counter_buff_, GpuFormat::R32_Uint);

            {
                const ShaderInfo shaders[] = {
                    {FaceIdVs_shader, 1, 0, 0},
                    {FaceIdPs_shader, 0, 0, 0},
                };

                const GpuFormat rtv_formats[] = {face_id_tex_.Format()};

                GpuRenderPipeline::States states;
                states.cull_mode = GpuRenderPipeline::CullMode::ClockWise;
                states.conservative_raster = true;
                states.depth_enable = true;
                states.rtv_formats = rtv_formats;
                states.dsv_format = ds_tex_.Format();

                const GpuVertexAttribs vertex_attribs(std::span<const GpuVertexAttrib>({
                    {"POSITION", 0, GpuFormat::RGB32_Float},
                }));

                render_pipeline_ =
                    GpuRenderPipeline(gpu_system_, GpuRenderPipeline::PrimitiveTopology::TriangleList, shaders, vertex_attribs, {}, states);
            }

            {
                mark_faces_cb_ = ConstantBuffer<MarkFacesConstantBuffer>(gpu_system_, L"mark_faces_cb_");

                const ShaderInfo shader = {MarkFacesCs_shader, 1, 1, 1};
                mark_faces_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                accum_faces_cb_ = ConstantBuffer<AccumFacesConstantBuffer>(gpu_system_, L"accum_faces_cb_");

                const ShaderInfo shader = {AccumFacesCs_shader, 1, 1, 1};
                accum_faces_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                filter_faces_cb_ = ConstantBuffer<FilterFacesConstantBuffer>(gpu_system_, L"filter_faces_cb_");

                const ShaderInfo shader = {FilterFacesCs_shader, 1, 2, 2};
                filter_faces_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
        }

        Mesh Process(const Mesh& mesh)
        {
            const auto& vertex_desc = mesh.MeshVertexDesc();
            const uint32_t vertex_stride = vertex_desc.Stride();

            GpuBuffer vb(gpu_system_, static_cast<uint32_t>(mesh.VertexBuffer().size() * sizeof(float)), GpuHeap::Upload,
                GpuResourceFlag::None, L"vb");
            std::memcpy(vb.Map(), mesh.VertexBuffer().data(), vb.Size());
            vb.Unmap(GpuRange{0, vb.Size()});

            GpuBuffer ib(gpu_system_, static_cast<uint32_t>(mesh.IndexBuffer().size() * sizeof(uint32_t)), GpuHeap::Upload,
                GpuResourceFlag::None, L"ib");
            std::memcpy(ib.Map(), mesh.IndexBuffer().data(), ib.Size());
            ib.Unmap(GpuRange{0, ib.Size()});

            const uint32_t num_indices = static_cast<uint32_t>(mesh.IndexBuffer().size());
            const uint32_t num_faces = num_indices / 3;

            face_mark_buff_ = GpuBuffer(
                gpu_system_, num_faces * sizeof(uint32_t), GpuHeap::Default, GpuResourceFlag::UnorderedAccess, L"face_mark_buff_");
            face_mark_srv_ = GpuShaderResourceView(gpu_system_, face_mark_buff_, GpuFormat::R32_Uint);
            face_mark_uav_ = GpuUnorderedAccessView(gpu_system_, face_mark_buff_, GpuFormat::R32_Uint);

            view_counter_buff_ = GpuBuffer(
                gpu_system_, num_faces * sizeof(uint32_t), GpuHeap::Default, GpuResourceFlag::UnorderedAccess, L"view_counter_buff");
            view_counter_srv_ = GpuShaderResourceView(gpu_system_, view_counter_buff_, GpuFormat::R32_Uint);
            view_counter_uav_ = GpuUnorderedAccessView(gpu_system_, view_counter_buff_, GpuFormat::R32_Uint);

            {
                mark_faces_cb_->width_height = glm::uvec2(face_id_tex_.Width(0), face_id_tex_.Height(0));
                mark_faces_cb_.UploadToGpu();

                accum_faces_cb_->num_faces = num_faces;
                accum_faces_cb_.UploadToGpu();

                filter_faces_cb_->num_faces = num_faces;
                filter_faces_cb_->threshold = NumViews / 100;
                filter_faces_cb_.UploadToGpu();
            }

            GpuCommandList cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
            {
                const uint32_t clear_clr[] = {0, 0, 0, 0};
                cmd_list.Clear(view_counter_uav_, clear_clr);
                cmd_list.Clear(filtered_counter_uav_, clear_clr);
            }

            for (uint32_t i = 0; i < NumViews; ++i)
            {
                const glm::vec2 angle = SphereHammersleySequence(i, NumViews);

                RenderFaceId(cmd_list, vb, vertex_stride, ib, num_indices, angle.x, angle.y, CameraDist);
                AccumulateFaces(cmd_list, num_faces);

                gpu_system_.ExecuteAndReset(cmd_list);
            }

            GpuBuffer filtered_index_buff(
                gpu_system_, num_indices * sizeof(uint32_t), GpuHeap::Default, GpuResourceFlag::UnorderedAccess, L"filtered_index_buff");
            FilterFaces(cmd_list, ib, num_faces, filtered_index_buff);

            auto filtered_indices = std::make_unique<uint32_t[]>(num_indices);
            const auto index_rb_future =
                cmd_list.ReadBackAsync(filtered_index_buff, filtered_indices.get(), num_indices * sizeof(uint32_t));

            uint32_t filtered_count = 0;
            const auto count_rb_future = cmd_list.ReadBackAsync(filtered_counter_buff_, &filtered_count, sizeof(filtered_count));

            gpu_system_.Execute(std::move(cmd_list));

            index_rb_future.wait();
            count_rb_future.wait();

            const uint32_t* filtered_indices_ptr = filtered_indices.get();
            return mesh.ExtractMesh(vertex_desc, {filtered_indices_ptr, filtered_indices_ptr + filtered_count * 3});
        }

    private:
        void RenderFaceId(GpuCommandList& cmd_list, const GpuBuffer& vb, uint32_t vertex_stride, const GpuBuffer& ib, uint32_t num_indices,
            float camera_azimuth, float camera_elevation, float camera_dist)
        {
            const glm::vec3 camera_pos = SphericalCameraPose(camera_azimuth, camera_elevation, camera_dist);
            const glm::vec3 camera_dir = -glm::normalize(camera_pos);
            glm::vec3 up_vec;
            if (std::abs(camera_dir.y) > 0.95f)
            {
                up_vec = glm::vec3(1, 0, 0);
            }
            else
            {
                up_vec = glm::vec3(0, 1, 0);
            }

            const glm::mat4x4 view_mtx = glm::lookAtRH(camera_pos, glm::vec3(0, 0, 0), up_vec);

            ConstantBuffer<RenderConstantBuffer> render_cb(gpu_system_, L"render_cb");
            render_cb->mvp = glm::transpose(proj_mtx_ * view_mtx);
            render_cb.UploadToGpu();

            const float clear_clr[] = {0, 0, 0, 0};
            cmd_list.Clear(face_id_rtv_, clear_clr);
            cmd_list.ClearDepth(dsv_, 1);

            const GpuCommandList::VertexBufferBinding vb_bindings[] = {{&vb, 0, vertex_stride}};
            const GpuCommandList::IndexBufferBinding ib_binding = {&ib, 0, GpuFormat::R32_Uint};

            const GeneralConstantBuffer* cbs[] = {&render_cb};
            const GpuCommandList::ShaderBinding shader_bindings[] = {
                {cbs, {}, {}},
                {{}, {}, {}},
            };

            const GpuRenderTargetView* rtvs[] = {&face_id_rtv_};

            const GpuViewport viewport = {0, 0, static_cast<float>(face_id_tex_.Width(0)), static_cast<float>(face_id_tex_.Height(0))};

            cmd_list.Render(
                render_pipeline_, vb_bindings, &ib_binding, num_indices, shader_bindings, rtvs, &dsv_, std::span(&viewport, 1), {});
        }

        void AccumulateFaces(GpuCommandList& cmd_list, uint32_t num_faces)
        {
            {
                constexpr uint32_t BlockDim = 16;

                const uint32_t clear_clr[] = {0, 0, 0, 0};
                cmd_list.Clear(face_mark_uav_, clear_clr);

                const GeneralConstantBuffer* cbs[] = {&mark_faces_cb_};
                const GpuShaderResourceView* srvs[] = {&face_id_srv_};
                GpuUnorderedAccessView* uavs[] = {&face_mark_uav_};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(mark_faces_pipeline_, DivUp(face_id_tex_.Width(0), BlockDim), DivUp(face_id_tex_.Height(0), BlockDim), 1,
                    shader_binding);
            }
            {
                constexpr uint32_t BlockDim = 256;

                const GeneralConstantBuffer* cbs[] = {&accum_faces_cb_};
                const GpuShaderResourceView* srvs[] = {&face_mark_srv_};
                GpuUnorderedAccessView* uavs[] = {&view_counter_uav_};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(accum_faces_pipeline_, DivUp(num_faces, BlockDim), 1, 1, shader_binding);
            }
        }

        void FilterFaces(GpuCommandList& cmd_list, const GpuBuffer& index_buff, uint32_t num_faces, GpuBuffer& filtered_index_buff)
        {
            constexpr uint32_t BlockDim = 256;

            GpuShaderResourceView index_srv(gpu_system_, index_buff, GpuFormat::R32_Uint);
            GpuUnorderedAccessView filtered_index_uav(gpu_system_, filtered_index_buff, GpuFormat::R32_Uint);

            const GeneralConstantBuffer* cbs[] = {&filter_faces_cb_};
            const GpuShaderResourceView* srvs[] = {&index_srv, &view_counter_srv_};
            GpuUnorderedAccessView* uavs[] = {&filtered_index_uav, &filtered_counter_uav_};
            const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
            cmd_list.Compute(filter_faces_pipeline_, DivUp(num_faces, BlockDim), 1, 1, shader_binding);
        }

    private:
        GpuSystem& gpu_system_;

        static constexpr uint32_t RtSize = 1024;
        static constexpr float CameraDist = 2;
        static constexpr uint32_t NumViews = 200;
        static constexpr float Fov = glm::radians(45.0f);

        GpuTexture2D face_id_tex_;
        GpuRenderTargetView face_id_rtv_;
        GpuShaderResourceView face_id_srv_;

        GpuTexture2D ds_tex_;
        GpuDepthStencilView dsv_;

        GpuBuffer face_mark_buff_;
        GpuShaderResourceView face_mark_srv_;
        GpuUnorderedAccessView face_mark_uav_;

        GpuBuffer view_counter_buff_;
        GpuShaderResourceView view_counter_srv_;
        GpuUnorderedAccessView view_counter_uav_;

        GpuBuffer filtered_counter_buff_;
        GpuUnorderedAccessView filtered_counter_uav_;

        glm::mat4x4 proj_mtx_;

        struct RenderConstantBuffer
        {
            glm::mat4x4 mvp;
        };
        GpuRenderPipeline render_pipeline_;

        struct MarkFacesConstantBuffer
        {
            glm::uvec2 width_height;
            uint32_t padding[2];
        };
        ConstantBuffer<MarkFacesConstantBuffer> mark_faces_cb_;
        GpuComputePipeline mark_faces_pipeline_;

        struct AccumFacesConstantBuffer
        {
            uint32_t num_faces;
            uint32_t padding[3];
        };
        ConstantBuffer<AccumFacesConstantBuffer> accum_faces_cb_;
        GpuComputePipeline accum_faces_pipeline_;

        struct FilterFacesConstantBuffer
        {
            uint32_t num_faces;
            uint32_t threshold;
            uint32_t padding[2];
        };
        ConstantBuffer<FilterFacesConstantBuffer> filter_faces_cb_;
        GpuComputePipeline filter_faces_pipeline_;
    };

    InvisibleFacesRemover::InvisibleFacesRemover(AIHoloImagerInternal& aihi) : impl_(std::make_unique<Impl>(aihi))
    {
    }

    InvisibleFacesRemover::~InvisibleFacesRemover() noexcept = default;

    InvisibleFacesRemover::InvisibleFacesRemover(InvisibleFacesRemover&& other) noexcept = default;
    InvisibleFacesRemover& InvisibleFacesRemover::operator=(InvisibleFacesRemover&& other) noexcept = default;

    Mesh InvisibleFacesRemover::Process(const Mesh& mesh)
    {
        return impl_->Process(mesh);
    }
} // namespace AIHoloImager
