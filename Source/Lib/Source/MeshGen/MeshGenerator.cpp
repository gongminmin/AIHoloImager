// Copyright (c) 2024 Minmin Gong
//

#include "MeshGenerator.hpp"

#include <array>
#include <cassert>
#include <iostream>
#include <set>

namespace AIHoloImager
{
    class MeshGenerator::Impl
    {
    public:
        Impl(const std::filesystem::path& exe_dir, PythonSystem& python_system) : python_system_(python_system)
        {
            mesh_generator_module_ = python_system_.Import("MeshGenerator");
            mesh_generator_class_ = python_system_.GetAttr(*mesh_generator_module_, "MeshGenerator");
            mesh_generator_ = python_system_.CallObject(*mesh_generator_class_);
            mesh_generator_gen_method_ = python_system_.GetAttr(*mesh_generator_, "Gen");

            // Copy the binary to exe's directory for future usage
            if (!std::filesystem::exists(exe_dir / "nvdiffrast_plugin.pyd"))
            {
                const std::filesystem::path local_app_data_dir = std::getenv("LOCALAPPDATA");
                const std::filesystem::path nvdiffrast_plugin_dir =
                    local_app_data_dir / ("torch_extensions/torch_extensions/Cache/py" AIHI_PY_VERSION "_cu121/nvdiffrast_plugin");
                std::filesystem::copy(nvdiffrast_plugin_dir / "nvdiffrast_plugin.pyd", exe_dir);
            }

            pil_module_ = python_system_.Import("PIL");
            image_class_ = python_system_.GetAttr(*pil_module_, "Image");
            image_frombuffer_method_ = python_system_.GetAttr(*image_class_, "frombuffer");
        }

        Mesh Generate(std::span<const Texture> input_images, uint32_t texture_size, const std::filesystem::path& tmp_dir)
        {
            assert(input_images.size() == 6);
            assert(input_images[0].Width() == 320);
            assert(input_images[0].Height() == 320);
            assert(input_images[0].NumChannels() == 3);

            PyObjectPtr py_input_images[6];
            for (size_t i = 0; i < input_images.size(); ++i)
            {
                auto& input_image = input_images[i];

                auto args = python_system_.MakeTuple(3);
                {
                    python_system_.SetTupleItem(*args, 0, python_system_.MakeObject(L"RGB"));
                }
                {
                    auto size = python_system_.MakeTuple(2);
                    {
                        python_system_.SetTupleItem(*size, 0, python_system_.MakeObject(input_image.Width()));
                        python_system_.SetTupleItem(*size, 1, python_system_.MakeObject(input_image.Height()));
                    }
                    python_system_.SetTupleItem(*args, 1, std::move(size));
                }
                {
                    auto image = python_system_.MakeObject(
                        std::span<const std::byte>(reinterpret_cast<const std::byte*>(input_image.Data()), input_image.DataSize()));
                    python_system_.SetTupleItem(*args, 2, std::move(image));
                }

                py_input_images[i] = python_system_.CallObject(*image_frombuffer_method_, *args);
            }

            auto args = python_system_.MakeTuple(3);
            {
                auto imgs_args = python_system_.MakeTuple(std::size(py_input_images));
                for (uint32_t i = 0; i < std::size(py_input_images); ++i)
                {
                    python_system_.SetTupleItem(*imgs_args, i, std::move(py_input_images[i]));
                }
                python_system_.SetTupleItem(*args, 0, std::move(imgs_args));
            }
            {
                python_system_.SetTupleItem(*args, 1, python_system_.MakeObject(texture_size));
            }

            std::filesystem::path output_mesh_path;
            {
                auto output_dir = tmp_dir / "Mesh";
                std::filesystem::create_directories(output_dir);
                output_mesh_path = output_dir / "Temp.obj";
                python_system_.SetTupleItem(*args, 2, python_system_.MakeObject(output_mesh_path.wstring()));
            }

            std::cout << "Generating mesh by AI...\n";
            python_system_.CallObject(*mesh_generator_gen_method_, *args);

            Mesh ai_mesh = LoadMesh(output_mesh_path);

            // The mesh from obj format is a triangle soup. Need to join closest vertices to make it a real mesh.

            Mesh joined_mesh(0, static_cast<uint32_t>(ai_mesh.Indices().size()));
            joined_mesh.AlbedoTexture(ai_mesh.AlbedoTexture());

            constexpr float Scale = 1e5f;

            std::set<std::array<int32_t, 5>> unique_int_vertex;
            for (uint32_t i = 0; i < ai_mesh.Vertices().size(); ++i)
            {
                const auto& vertex = ai_mesh.Vertex(i);
                std::array<int32_t, 5> int_vertex = {static_cast<int32_t>(vertex.pos.x * Scale + 0.5f),
                    static_cast<int32_t>(vertex.pos.y * Scale + 0.5f), static_cast<int32_t>(vertex.pos.z * Scale + 0.5f),
                    static_cast<int32_t>(vertex.texcoord.x * Scale + 0.5f), static_cast<int32_t>(vertex.texcoord.y * Scale + 0.5f)};
                unique_int_vertex.emplace(std::move(int_vertex));
            }

            std::vector<std::array<int32_t, 5>> unique_int_vertex_vec(unique_int_vertex.begin(), unique_int_vertex.end());

            joined_mesh.ResizeVertices(static_cast<uint32_t>(unique_int_vertex_vec.size()));
            std::vector<uint32_t> unique_vertex_mapping(ai_mesh.Vertices().size());

#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < ai_mesh.Vertices().size(); ++i)
            {
                const auto& vertex = ai_mesh.Vertex(i);
                const std::array<int32_t, 5> int_vertex = {static_cast<int32_t>(vertex.pos.x * Scale + 0.5f),
                    static_cast<int32_t>(vertex.pos.y * Scale + 0.5f), static_cast<int32_t>(vertex.pos.z * Scale + 0.5f),
                    static_cast<int32_t>(vertex.texcoord.x * Scale + 0.5f), static_cast<int32_t>(vertex.texcoord.y * Scale + 0.5f)};

                const auto iter = std::lower_bound(unique_int_vertex_vec.begin(), unique_int_vertex_vec.end(), int_vertex);
                assert(*iter == int_vertex);

                const uint32_t found_index = static_cast<uint32_t>(iter - unique_int_vertex_vec.begin());
                joined_mesh.Vertex(found_index) = vertex;
                unique_vertex_mapping[i] = found_index;
            }

            for (uint32_t i = 0; i < static_cast<uint32_t>(ai_mesh.Indices().size()); ++i)
            {
                joined_mesh.Index(i) = unique_vertex_mapping[ai_mesh.Index(i)];
            }

            return joined_mesh;
        }

    private:
        PythonSystem& python_system_;

        PyObjectPtr mesh_generator_module_;
        PyObjectPtr mesh_generator_class_;
        PyObjectPtr mesh_generator_;
        PyObjectPtr mesh_generator_gen_method_;

        PyObjectPtr pil_module_;
        PyObjectPtr image_class_;
        PyObjectPtr image_frombuffer_method_;
    };

    MeshGenerator::MeshGenerator(const std::filesystem::path& exe_dir, PythonSystem& python_system)
        : impl_(std::make_unique<Impl>(exe_dir, python_system))
    {
    }

    MeshGenerator::~MeshGenerator() noexcept = default;

    MeshGenerator::MeshGenerator(MeshGenerator&& other) noexcept = default;
    MeshGenerator& MeshGenerator::operator=(MeshGenerator&& other) noexcept = default;

    Mesh MeshGenerator::Generate(std::span<const Texture> input_images, uint32_t texture_size, const std::filesystem::path& tmp_dir)
    {
        return impl_->Generate(std::move(input_images), texture_size, tmp_dir);
    }
} // namespace AIHoloImager
