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
                diff_optimizer_ = python_system.CallObject(*diff_optimizer_class_, reinterpret_cast<void*>(&gpu_system));
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

        void OptimizeTransform(const Mesh& mesh, glm::mat4x4& model_mtx, std::span<const AIHoloImagerInternal::ProjectionDesc> projections)
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

            const uint32_t num_images = static_cast<uint32_t>(projections.size());
            std::vector<glm::mat4x4> view_proj_mtxs(num_images);
            std::vector<glm::vec2> vp_offsets(num_images);
            for (uint32_t i = 0; i < num_images; ++i)
            {
                const auto& projection = projections[i];

                view_proj_mtxs[i] = projection.proj_mtx * projection.view_mtx;
                vp_offsets[i] = projection.vp_offset;
            }

            {
                PerfRegion wait_perf(aihi_.PerfProfilerInstance(), "Wait for init");
                py_init_future_.wait();
            }

            {
                PythonSystem::GilGuard guard;

                auto& gpu_system = aihi_.GpuSystemInstance();
                auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

                auto& python_system = aihi_.PythonSystemInstance();

                auto imgs_args = python_system.MakeTupleOfSize(num_images);
                for (uint32_t i = 0; i < num_images; ++i)
                {
                    const auto& projection = projections[i];

                    const auto& delighted_tex = *projection.image;
                    PyObjectPtr image = MakePyObjectPtr(tensor_converter.ConvertPy(cmd_list, delighted_tex));
                    auto img_tuple = python_system.MakeTuple(std::move(image), projection.image_offset.x, projection.image_offset.y,
                        projection.full_width, projection.full_height);
                    python_system.SetTupleItem(*imgs_args, i, std::move(img_tuple));
                }
                gpu_system.Execute(std::move(cmd_list));

                const auto indices = mesh.IndexBuffer();
                auto py_opt_transforms = python_system.CallObject(*diff_optimizer_opt_transform_method_,
                    std::span(reinterpret_cast<const std::byte*>(positions.data()), positions.size() * sizeof(glm::vec3)),
                    std::span(reinterpret_cast<const std::byte*>(colors.data()), colors.size() * sizeof(glm::vec3)),
                    static_cast<uint32_t>(positions.size()),
                    std::span(reinterpret_cast<const std::byte*>(indices.data()), indices.size() * sizeof(uint32_t)),
                    static_cast<uint32_t>(indices.size()), std::move(imgs_args),
                    std::span(reinterpret_cast<const std::byte*>(view_proj_mtxs.data()), view_proj_mtxs.size() * sizeof(glm::mat4x4)),
                    std::span(reinterpret_cast<const std::byte*>(vp_offsets.data()), vp_offsets.size() * sizeof(glm::vec2)), num_images,
                    std::span(reinterpret_cast<const std::byte*>(&scale), sizeof(scale)),
                    std::span(reinterpret_cast<const std::byte*>(&rotation), sizeof(rotation)),
                    std::span(reinterpret_cast<const std::byte*>(&translation), sizeof(translation)));

                const auto scale_opt = python_system.ToSpan<const float>(*python_system.GetTupleItem(*py_opt_transforms, 0));
                const auto rotate_opt = python_system.ToSpan<const float>(*python_system.GetTupleItem(*py_opt_transforms, 1));
                const auto translate_opt = python_system.ToSpan<const float>(*python_system.GetTupleItem(*py_opt_transforms, 2));
                scale = glm::vec3(scale_opt[0], scale_opt[1], scale_opt[2]);
                rotation = glm::quat(rotate_opt[3], rotate_opt[0], rotate_opt[1], rotate_opt[2]);
                translation = glm::vec3(translate_opt[0], translate_opt[1], translate_opt[2]);
                model_mtx = glm::recompose(scale, rotation, translation, skew, perspective);
            }
        }

        void OptimizeTexture(Mesh& mesh, const glm::mat4x4& model_mtx, std::span<const AIHoloImagerInternal::ProjectionDesc> projections,
            GpuTexture2D& albedo_tex, const GpuTexture2D& mask_tex)
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

            const uint32_t num_images = static_cast<uint32_t>(projections.size());
            std::vector<glm::mat4x4> mvp_mtxs(num_images);
            std::vector<glm::vec2> vp_offsets(num_images);
            for (uint32_t i = 0; i < num_images; ++i)
            {
                const auto& projection = projections[i];

                mvp_mtxs[i] = projection.proj_mtx * projection.view_mtx * model_mtx;
                vp_offsets[i] = projection.vp_offset;
            }

            {
                PerfRegion wait_perf(aihi_.PerfProfilerInstance(), "Wait for init");
                py_init_future_.wait();
            }

            {
                PythonSystem::GilGuard guard;

                auto& gpu_system = aihi_.GpuSystemInstance();
                auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

                auto& python_system = aihi_.PythonSystemInstance();

                auto imgs_args = python_system.MakeTupleOfSize(num_images);
                for (uint32_t i = 0; i < num_images; ++i)
                {
                    const auto& projection = projections[i];

                    const auto& delighted_tex = *projection.image;
                    PyObjectPtr image = MakePyObjectPtr(tensor_converter.ConvertPy(cmd_list, delighted_tex));
                    auto img_tuple = python_system.MakeTuple(std::move(image), projection.image_offset.x, projection.image_offset.y,
                        projection.full_width, projection.full_height);
                    python_system.SetTupleItem(*imgs_args, i, std::move(img_tuple));
                }
                PyObjectPtr py_albedo_img = MakePyObjectPtr(tensor_converter.ConvertPy(cmd_list, albedo_tex));
                PyObjectPtr py_mask_img = MakePyObjectPtr(tensor_converter.ConvertPy(cmd_list, mask_tex));
                gpu_system.ExecuteAndReset(cmd_list);

                const auto indices = mesh.IndexBuffer();
                auto py_opt_texture = python_system.CallObject(*diff_optimizer_opt_texture_method_,
                    std::span(reinterpret_cast<const std::byte*>(positions.data()), positions.size() * sizeof(glm::vec3)),
                    std::span(reinterpret_cast<const std::byte*>(tex_coords.data()), tex_coords.size() * sizeof(glm::vec2)),
                    static_cast<uint32_t>(positions.size()),
                    std::span(reinterpret_cast<const std::byte*>(indices.data()), indices.size() * sizeof(uint32_t)),
                    static_cast<uint32_t>(indices.size()), std::move(imgs_args),
                    std::span(reinterpret_cast<const std::byte*>(mvp_mtxs.data()), mvp_mtxs.size() * sizeof(glm::mat4x4)),
                    std::span(reinterpret_cast<const std::byte*>(vp_offsets.data()), vp_offsets.size() * sizeof(glm::vec2)), num_images,
                    std::move(py_albedo_img), std::move(py_mask_img));

                tensor_converter.ConvertPy(
                    cmd_list, *py_opt_texture, albedo_tex, GpuFormat::RGBA8_UNorm, GpuResourceFlag::None, "albedo_tex");
                gpu_system.Execute(std::move(cmd_list));
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

    void DiffOptimizer::OptimizeTransform(
        const Mesh& mesh, glm::mat4x4& model_mtx, std::span<const AIHoloImagerInternal::ProjectionDesc> projections)
    {
        impl_->OptimizeTransform(mesh, model_mtx, std::move(projections));
    }

    void DiffOptimizer::OptimizeTexture(Mesh& mesh, const glm::mat4x4& model_mtx,
        std::span<const AIHoloImagerInternal::ProjectionDesc> projections, GpuTexture2D& albedo_tex, const GpuTexture2D& mask_tex)
    {
        impl_->OptimizeTexture(mesh, model_mtx, std::move(projections), albedo_tex, mask_tex);
    }
} // namespace AIHoloImager
