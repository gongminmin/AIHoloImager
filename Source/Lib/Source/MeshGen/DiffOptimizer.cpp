// Copyright (c) 2024-2025 Minmin Gong
//

#include "DiffOptimizer.hpp"

#include <span>

#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#ifndef GLM_ENABLE_EXPERIMENTAL
    #define GLM_ENABLE_EXPERIMENTAL
#endif
#include <glm/gtx/matrix_decompose.hpp>

#include "Gpu/GpuSystem.hpp"

namespace AIHoloImager
{
    class DiffOptimizer::Impl
    {
    public:
        Impl(GpuSystem& gpu_system, PythonSystem& python_system) : gpu_system_(gpu_system), python_system_(python_system)
        {
            diff_optimizer_module_ = python_system_.Import("DiffOptimizer");
            diff_optimizer_class_ = python_system_.GetAttr(*diff_optimizer_module_, "DiffOptimizer");
            auto args = python_system_.MakeTuple(1);
            {
                python_system_.SetTupleItem(*args, 0, python_system_.MakeObject(reinterpret_cast<void*>(&gpu_system_)));
            }
            diff_optimizer_ = python_system_.CallObject(*diff_optimizer_class_, *args);
            diff_optimizer_opt_method_ = python_system_.GetAttr(*diff_optimizer_, "Optimize");
        }

        ~Impl()
        {
            auto diff_optimizer_destroy_method = python_system_.GetAttr(*diff_optimizer_, "Destroy");
            python_system_.CallObject(*diff_optimizer_destroy_method);
        }

        void Optimize(Mesh& mesh, glm::mat4x4& model_mtx, const StructureFromMotion::Result& sfm_input)
        {
            glm::vec3 scale;
            glm::quat rotation;
            glm::vec3 translation;
            glm::vec3 skew;
            glm::vec4 perspective;
            glm::decompose(model_mtx, scale, rotation, translation, skew, perspective);

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

            auto args = python_system_.MakeTuple(12);
            {
                python_system_.SetTupleItem(*args, 0,
                    python_system_.MakeObject(
                        std::span(reinterpret_cast<const std::byte*>(positions.data()), positions.size() * sizeof(glm::vec3))));
                python_system_.SetTupleItem(*args, 1,
                    python_system_.MakeObject(
                        std::span(reinterpret_cast<const std::byte*>(colors.data()), colors.size() * sizeof(glm::vec3))));
                python_system_.SetTupleItem(*args, 2, python_system_.MakeObject(static_cast<uint32_t>(positions.size())));

                const auto indices = mesh.IndexBuffer();
                python_system_.SetTupleItem(*args, 3,
                    python_system_.MakeObject(
                        std::span(reinterpret_cast<const std::byte*>(indices.data()), indices.size() * sizeof(uint32_t))));
                python_system_.SetTupleItem(*args, 4, python_system_.MakeObject(static_cast<uint32_t>(indices.size())));

                auto imgs_args = python_system_.MakeTuple(num_images);
                for (uint32_t i = 0; i < num_images; ++i)
                {
                    auto img_tuple = python_system_.MakeTuple(7);

                    const auto& delighted_image = sfm_input.views[i].delighted_image;
                    auto image = python_system_.MakeObject(
                        std::span(reinterpret_cast<const std::byte*>(delighted_image.Data()), delighted_image.DataSize()));
                    python_system_.SetTupleItem(*img_tuple, 0, std::move(image));
                    python_system_.SetTupleItem(*img_tuple, 1, python_system_.MakeObject(sfm_input.views[i].delighted_offset.x));
                    python_system_.SetTupleItem(*img_tuple, 2, python_system_.MakeObject(sfm_input.views[i].delighted_offset.y));
                    python_system_.SetTupleItem(*img_tuple, 3, python_system_.MakeObject(delighted_image.Width()));
                    python_system_.SetTupleItem(*img_tuple, 4, python_system_.MakeObject(delighted_image.Height()));
                    python_system_.SetTupleItem(*img_tuple, 5, python_system_.MakeObject(sfm_input.views[i].image_mask.Width()));
                    python_system_.SetTupleItem(*img_tuple, 6, python_system_.MakeObject(sfm_input.views[i].image_mask.Height()));

                    python_system_.SetTupleItem(*imgs_args, i, std::move(img_tuple));
                }
                python_system_.SetTupleItem(*args, 5, std::move(imgs_args));

                python_system_.SetTupleItem(*args, 6,
                    python_system_.MakeObject(
                        std::span(reinterpret_cast<const std::byte*>(view_proj_mtxs.data()), view_proj_mtxs.size() * sizeof(glm::mat4x4))));
                python_system_.SetTupleItem(*args, 7,
                    python_system_.MakeObject(std::span(
                        reinterpret_cast<const std::byte*>(transform_offsets.data()), transform_offsets.size() * sizeof(glm::ivec2))));

                python_system_.SetTupleItem(*args, 8, python_system_.MakeObject(num_images));

                python_system_.SetTupleItem(
                    *args, 9, python_system_.MakeObject(std::span(reinterpret_cast<const std::byte*>(&scale), sizeof(scale))));
                python_system_.SetTupleItem(
                    *args, 10, python_system_.MakeObject(std::span(reinterpret_cast<const std::byte*>(&rotation), sizeof(rotation))));
                python_system_.SetTupleItem(
                    *args, 11, python_system_.MakeObject(std::span(reinterpret_cast<const std::byte*>(&translation), sizeof(translation))));
            }

            auto py_opt_transforms = python_system_.CallObject(*diff_optimizer_opt_method_, *args);

            const auto scale_opt = python_system_.ToSpan<const float>(*python_system_.GetTupleItem(*py_opt_transforms, 0));
            const auto rotate_opt = python_system_.ToSpan<const float>(*python_system_.GetTupleItem(*py_opt_transforms, 1));
            const auto translate_opt = python_system_.ToSpan<const float>(*python_system_.GetTupleItem(*py_opt_transforms, 2));
            scale = glm::vec3(scale_opt[0], scale_opt[1], scale_opt[2]);
            rotation = glm::quat(rotate_opt[3], rotate_opt[0], rotate_opt[1], rotate_opt[2]);
            translation = glm::vec3(translate_opt[0], translate_opt[1], translate_opt[2]);
            model_mtx = glm::recompose(scale, rotation, translation, skew, perspective);
        }

    private:
        GpuSystem& gpu_system_;
        PythonSystem& python_system_;

        PyObjectPtr diff_optimizer_module_;
        PyObjectPtr diff_optimizer_class_;
        PyObjectPtr diff_optimizer_;
        PyObjectPtr diff_optimizer_opt_method_;
    };

    DiffOptimizer::DiffOptimizer(GpuSystem& gpu_system, PythonSystem& python_system)
        : impl_(std::make_unique<Impl>(gpu_system, python_system))
    {
    }

    DiffOptimizer::~DiffOptimizer() noexcept = default;

    DiffOptimizer::DiffOptimizer(DiffOptimizer&& other) noexcept = default;
    DiffOptimizer& DiffOptimizer::operator=(DiffOptimizer&& other) noexcept = default;

    void DiffOptimizer::Optimize(Mesh& mesh, glm::mat4x4& model_mtx, const StructureFromMotion::Result& sfm_input)
    {
        impl_->Optimize(mesh, model_mtx, sfm_input);
    }
} // namespace AIHoloImager
