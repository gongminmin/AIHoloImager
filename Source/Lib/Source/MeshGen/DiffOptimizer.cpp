// Copyright (c) 2024-2026 Minmin Gong
//

#include "DiffOptimizer.hpp"

#include <future>
#include <span>

#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#ifndef GLM_ENABLE_EXPERIMENTAL
    #define GLM_ENABLE_EXPERIMENTAL
#endif
#include <glm/gtx/matrix_decompose.hpp>

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuSystem.hpp"

namespace AIHoloImager
{
    class DiffOptimizer::Impl
    {
    public:
        explicit Impl(AIHoloImagerInternal& aihi) : aihi_(aihi)
        {
            py_init_future_ = std::async(std::launch::async, [this] {
                PerfRegion init_async_perf(aihi_.PerfProfilerInstance(), "DiffOptimizer init (async)");

                PythonSystem::GilGuard guard;

                auto& gpu_system = aihi_.GpuSystemInstance();
                auto& python_system = aihi_.PythonSystemInstance();

                diff_optimizer_module_ = python_system.Import("DiffOptimizer");
                diff_optimizer_class_ = python_system.GetAttr(*diff_optimizer_module_, "DiffOptimizer");
                auto args = python_system.MakeTuple(1);
                {
                    python_system.SetTupleItem(*args, 0, python_system.MakeObject(reinterpret_cast<void*>(&gpu_system)));
                }
                diff_optimizer_ = python_system.CallObject(*diff_optimizer_class_, *args);
                diff_optimizer_opt_transform_method_ = python_system.GetAttr(*diff_optimizer_, "OptimizeTransform");
                diff_optimizer_opt_texture_method_ = python_system.GetAttr(*diff_optimizer_, "OptimizeTexture");
            });
        }

        ~Impl()
        {
            PerfRegion destroy_perf(aihi_.PerfProfilerInstance(), "DiffOptimizer destroy");

            py_init_future_.wait();

            PythonSystem::GilGuard guard;

            auto& python_system = aihi_.PythonSystemInstance();
            auto diff_optimizer_destroy_method = python_system.GetAttr(*diff_optimizer_, "Destroy");
            python_system.CallObject(*diff_optimizer_destroy_method);

            diff_optimizer_destroy_method.reset();
            diff_optimizer_opt_texture_method_.reset();
            diff_optimizer_opt_transform_method_.reset();
            diff_optimizer_.reset();
            diff_optimizer_class_.reset();
            diff_optimizer_module_.reset();
        }

        void OptimizeTransform(const Mesh& mesh, glm::mat4x4& model_mtx, const StructureFromMotion::Result& sfm_input)
        {
            glm::vec3 scale;
            glm::quat rotation;
            glm::vec3 translation;
            glm::vec3 skew;
            glm::vec4 perspective;
            glm::decompose(model_mtx, scale, rotation, translation, skew, perspective);

            auto& tensor_converter = aihi_.TensorConverterInstance();

            const auto& vertex_desc = mesh.MeshVertexDesc();
            const uint32_t pos_attrib_index = vertex_desc.FindAttrib(VertexAttrib::Semantic::Position, 0);
            const uint32_t color_attrib_index = vertex_desc.FindAttrib(VertexAttrib::Semantic::Color, 0);

            std::vector<glm::vec3> positions(mesh.NumVertices());
            std::vector<glm::vec3> colors(mesh.NumVertices());
#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < mesh.NumVertices(); ++i)
            {
                positions[i] = mesh.VertexData<glm::vec3>(i, pos_attrib_index);
                colors[i] = mesh.VertexData<glm::vec3>(i, color_attrib_index);
            }

            const uint32_t num_images = static_cast<uint32_t>(sfm_input.views.size());
            std::vector<glm::mat4x4> view_proj_mtxs(num_images);
            std::vector<glm::ivec2> transform_offsets(num_images);
            for (uint32_t i = 0; i < num_images; ++i)
            {
                const auto& view = sfm_input.views[i];
                const auto& intrinsic = sfm_input.intrinsics[view.intrinsic_id];

                const glm::mat4x4 view_mtx = CalcViewMatrix(view);
                const glm::mat4x4 proj_mtx = CalcProjMatrix(intrinsic, 0.1f, 30.0f);

                view_proj_mtxs[i] = proj_mtx * view_mtx;

                transform_offsets[i] = {
                    intrinsic.k[0].z - intrinsic.width / 2,
                    intrinsic.k[1].z - intrinsic.height / 2,
                };
            }

            {
                PerfRegion wait_perf(aihi_.PerfProfilerInstance(), "Wait for init");
                py_init_future_.wait();
            }

            {
                PythonSystem::GilGuard guard;

                auto& gpu_system = aihi_.GpuSystemInstance();
                auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Compute);

                auto& python_system = aihi_.PythonSystemInstance();
                auto args = python_system.MakeTuple(12);
                {
                    python_system.SetTupleItem(*args, 0,
                        python_system.MakeObject(
                            std::span(reinterpret_cast<const std::byte*>(positions.data()), positions.size() * sizeof(glm::vec3))));
                    python_system.SetTupleItem(*args, 1,
                        python_system.MakeObject(
                            std::span(reinterpret_cast<const std::byte*>(colors.data()), colors.size() * sizeof(glm::vec3))));
                    python_system.SetTupleItem(*args, 2, python_system.MakeObject(static_cast<uint32_t>(positions.size())));

                    const auto indices = mesh.IndexBuffer();
                    python_system.SetTupleItem(*args, 3,
                        python_system.MakeObject(
                            std::span(reinterpret_cast<const std::byte*>(indices.data()), indices.size() * sizeof(uint32_t))));
                    python_system.SetTupleItem(*args, 4, python_system.MakeObject(static_cast<uint32_t>(indices.size())));

                    auto imgs_args = python_system.MakeTuple(num_images);
                    for (uint32_t i = 0; i < num_images; ++i)
                    {
                        auto img_tuple = python_system.MakeTuple(5);

                        const auto& view = sfm_input.views[i];
                        const auto& intrinsic = sfm_input.intrinsics[view.intrinsic_id];

                        const auto& delighted_tex = view.delighted_tex;
                        PyObjectPtr image = MakePyObjectPtr(tensor_converter.ConvertPy(cmd_list, delighted_tex));
                        python_system.SetTupleItem(*img_tuple, 0, std::move(image));
                        python_system.SetTupleItem(*img_tuple, 1, python_system.MakeObject(view.delighted_offset.x));
                        python_system.SetTupleItem(*img_tuple, 2, python_system.MakeObject(view.delighted_offset.y));
                        python_system.SetTupleItem(*img_tuple, 3, python_system.MakeObject(intrinsic.width));
                        python_system.SetTupleItem(*img_tuple, 4, python_system.MakeObject(intrinsic.height));

                        python_system.SetTupleItem(*imgs_args, i, std::move(img_tuple));
                    }
                    python_system.SetTupleItem(*args, 5, std::move(imgs_args));

                    python_system.SetTupleItem(*args, 6,
                        python_system.MakeObject(std::span(
                            reinterpret_cast<const std::byte*>(view_proj_mtxs.data()), view_proj_mtxs.size() * sizeof(glm::mat4x4))));
                    python_system.SetTupleItem(*args, 7,
                        python_system.MakeObject(std::span(
                            reinterpret_cast<const std::byte*>(transform_offsets.data()), transform_offsets.size() * sizeof(glm::ivec2))));

                    python_system.SetTupleItem(*args, 8, python_system.MakeObject(num_images));

                    python_system.SetTupleItem(
                        *args, 9, python_system.MakeObject(std::span(reinterpret_cast<const std::byte*>(&scale), sizeof(scale))));
                    python_system.SetTupleItem(
                        *args, 10, python_system.MakeObject(std::span(reinterpret_cast<const std::byte*>(&rotation), sizeof(rotation))));
                    python_system.SetTupleItem(*args, 11,
                        python_system.MakeObject(std::span(reinterpret_cast<const std::byte*>(&translation), sizeof(translation))));
                }
                gpu_system.Execute(std::move(cmd_list));

                auto py_opt_transforms = python_system.CallObject(*diff_optimizer_opt_transform_method_, *args);

                const auto scale_opt = python_system.ToSpan<const float>(*python_system.GetTupleItem(*py_opt_transforms, 0));
                const auto rotate_opt = python_system.ToSpan<const float>(*python_system.GetTupleItem(*py_opt_transforms, 1));
                const auto translate_opt = python_system.ToSpan<const float>(*python_system.GetTupleItem(*py_opt_transforms, 2));
                scale = glm::vec3(scale_opt[0], scale_opt[1], scale_opt[2]);
                rotation = glm::quat(rotate_opt[3], rotate_opt[0], rotate_opt[1], rotate_opt[2]);
                translation = glm::vec3(translate_opt[0], translate_opt[1], translate_opt[2]);
                model_mtx = glm::recompose(scale, rotation, translation, skew, perspective);
            }
        }

        void OptimizeTexture(
            Mesh& mesh, const glm::mat4x4& model_mtx, const StructureFromMotion::Result& sfm_input, const Texture& mask_tex)
        {
            auto& tensor_converter = aihi_.TensorConverterInstance();

            const auto& vertex_desc = mesh.MeshVertexDesc();
            const uint32_t pos_attrib_index = vertex_desc.FindAttrib(VertexAttrib::Semantic::Position, 0);
            const uint32_t tex_coord_attrib_index = vertex_desc.FindAttrib(VertexAttrib::Semantic::TexCoord, 0);

            std::vector<glm::vec3> positions(mesh.NumVertices());
            std::vector<glm::vec2> tex_coords(mesh.NumVertices());
#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < mesh.NumVertices(); ++i)
            {
                positions[i] = mesh.VertexData<glm::vec3>(i, pos_attrib_index);
                tex_coords[i] = mesh.VertexData<glm::vec2>(i, tex_coord_attrib_index);
                tex_coords[i].y = 1 - tex_coords[i].y;
            }

            const uint32_t num_images = static_cast<uint32_t>(sfm_input.views.size());
            std::vector<glm::mat4x4> mvp_mtxs(num_images);
            std::vector<glm::ivec2> transform_offsets(num_images);
            for (uint32_t i = 0; i < num_images; ++i)
            {
                const auto& view = sfm_input.views[i];
                const auto& intrinsic = sfm_input.intrinsics[view.intrinsic_id];

                const glm::mat4x4 view_mtx = CalcViewMatrix(view);
                const glm::mat4x4 proj_mtx = CalcProjMatrix(intrinsic, 0.1f, 30.0f);

                mvp_mtxs[i] = proj_mtx * view_mtx * model_mtx;

                transform_offsets[i] = {
                    intrinsic.k[0].z - intrinsic.width / 2,
                    intrinsic.k[1].z - intrinsic.height / 2,
                };
            }

            auto& texture = mesh.AlbedoTexture();

            {
                PerfRegion wait_perf(aihi_.PerfProfilerInstance(), "Wait for init");
                py_init_future_.wait();
            }

            {
                PythonSystem::GilGuard guard;

                auto& gpu_system = aihi_.GpuSystemInstance();
                auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Compute);

                auto& python_system = aihi_.PythonSystemInstance();
                auto args = python_system.MakeTuple(13);
                {
                    python_system.SetTupleItem(*args, 0,
                        python_system.MakeObject(
                            std::span(reinterpret_cast<const std::byte*>(positions.data()), positions.size() * sizeof(glm::vec3))));
                    python_system.SetTupleItem(*args, 1,
                        python_system.MakeObject(
                            std::span(reinterpret_cast<const std::byte*>(tex_coords.data()), tex_coords.size() * sizeof(glm::vec2))));
                    python_system.SetTupleItem(*args, 2, python_system.MakeObject(static_cast<uint32_t>(positions.size())));

                    const auto indices = mesh.IndexBuffer();
                    python_system.SetTupleItem(*args, 3,
                        python_system.MakeObject(
                            std::span(reinterpret_cast<const std::byte*>(indices.data()), indices.size() * sizeof(uint32_t))));
                    python_system.SetTupleItem(*args, 4, python_system.MakeObject(static_cast<uint32_t>(indices.size())));

                    auto imgs_args = python_system.MakeTuple(num_images);
                    for (uint32_t i = 0; i < num_images; ++i)
                    {
                        auto img_tuple = python_system.MakeTuple(5);

                        const auto& view = sfm_input.views[i];
                        const auto& intrinsic = sfm_input.intrinsics[view.intrinsic_id];

                        const auto& delighted_tex = view.delighted_tex;
                        PyObjectPtr image = MakePyObjectPtr(tensor_converter.ConvertPy(cmd_list, delighted_tex));
                        python_system.SetTupleItem(*img_tuple, 0, std::move(image));
                        python_system.SetTupleItem(*img_tuple, 1, python_system.MakeObject(view.delighted_offset.x));
                        python_system.SetTupleItem(*img_tuple, 2, python_system.MakeObject(view.delighted_offset.y));
                        python_system.SetTupleItem(*img_tuple, 3, python_system.MakeObject(intrinsic.width));
                        python_system.SetTupleItem(*img_tuple, 4, python_system.MakeObject(intrinsic.height));

                        python_system.SetTupleItem(*imgs_args, i, std::move(img_tuple));
                    }
                    python_system.SetTupleItem(*args, 5, std::move(imgs_args));

                    python_system.SetTupleItem(*args, 6,
                        python_system.MakeObject(
                            std::span(reinterpret_cast<const std::byte*>(mvp_mtxs.data()), mvp_mtxs.size() * sizeof(glm::mat4x4))));
                    python_system.SetTupleItem(*args, 7,
                        python_system.MakeObject(std::span(
                            reinterpret_cast<const std::byte*>(transform_offsets.data()), transform_offsets.size() * sizeof(glm::ivec2))));

                    python_system.SetTupleItem(*args, 8, python_system.MakeObject(num_images));

                    python_system.SetTupleItem(*args, 9,
                        python_system.MakeObject(std::span(reinterpret_cast<const std::byte*>(texture.Data()), texture.DataSize())));
                    python_system.SetTupleItem(*args, 10, python_system.MakeObject(static_cast<uint32_t>(texture.Width())));
                    python_system.SetTupleItem(*args, 11, python_system.MakeObject(static_cast<uint32_t>(texture.Height())));

                    python_system.SetTupleItem(*args, 12,
                        python_system.MakeObject(std::span(reinterpret_cast<const std::byte*>(mask_tex.Data()), mask_tex.DataSize())));
                }
                gpu_system.ExecuteAndReset(cmd_list);

                auto py_opt_texture = python_system.CallObject(*diff_optimizer_opt_texture_method_, *args);

                GpuTexture2D texture_opt;
                tensor_converter.ConvertPy(
                    cmd_list, *py_opt_texture, texture_opt, GpuFormat::RGBA8_UNorm, GpuResourceFlag::None, "texture_opt");
                const auto rb_future = cmd_list.ReadBackAsync(texture_opt, 0, texture.Data(), texture.DataSize());
                gpu_system.Execute(std::move(cmd_list));
                rb_future.wait();
            }
        }

    private:
        AIHoloImagerInternal& aihi_;

        PyObjectPtr diff_optimizer_module_;
        PyObjectPtr diff_optimizer_class_;
        PyObjectPtr diff_optimizer_;
        PyObjectPtr diff_optimizer_opt_transform_method_;
        PyObjectPtr diff_optimizer_opt_texture_method_;
        std::future<void> py_init_future_;
    };

    DiffOptimizer::DiffOptimizer(AIHoloImagerInternal& aihi) : impl_(std::make_unique<Impl>(aihi))
    {
    }

    DiffOptimizer::~DiffOptimizer() noexcept = default;

    DiffOptimizer::DiffOptimizer(DiffOptimizer&& other) noexcept = default;
    DiffOptimizer& DiffOptimizer::operator=(DiffOptimizer&& other) noexcept = default;

    void DiffOptimizer::OptimizeTransform(const Mesh& mesh, glm::mat4x4& model_mtx, const StructureFromMotion::Result& sfm_input)
    {
        impl_->OptimizeTransform(mesh, model_mtx, sfm_input);
    }

    void DiffOptimizer::OptimizeTexture(
        Mesh& mesh, const glm::mat4x4& model_mtx, const StructureFromMotion::Result& sfm_input, const Texture& mask_tex)
    {
        impl_->OptimizeTexture(mesh, model_mtx, sfm_input, mask_tex);
    }
} // namespace AIHoloImager
