// Copyright (c) 2024 Minmin Gong
//

#include "MeshGenerator.hpp"

#include <cassert>
#include <iostream>

namespace AIHoloImager
{
    class MeshGenerator::Impl
    {
    public:
        explicit Impl(PythonSystem& python_system) : python_system_(python_system)
        {
            mesh_generator_module_ = python_system_.Import("MeshGenerator");
            mesh_generator_class_ = python_system_.GetAttr(*mesh_generator_module_, "MeshGenerator");
            mesh_generator_ = python_system_.CallObject(*mesh_generator_class_);
            mesh_generator_gen_method_ = python_system_.GetAttr(*mesh_generator_, "Gen");

            pil_module_ = python_system_.Import("PIL");
            image_class_ = python_system_.GetAttr(*pil_module_, "Image");
            image_frombuffer_method_ = python_system_.GetAttr(*image_class_, "frombuffer");
        }

        Mesh Generate(std::span<const Texture> input_images, const std::filesystem::path& tmp_dir)
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

            auto args = python_system_.MakeTuple(2);
            {
                auto imgs_args = python_system_.MakeTuple(std::size(py_input_images));
                for (uint32_t i = 0; i < std::size(py_input_images); ++i)
                {
                    python_system_.SetTupleItem(*imgs_args, i, std::move(py_input_images[i]));
                }
                python_system_.SetTupleItem(*args, 0, std::move(imgs_args));
            }

            std::filesystem::path output_mesh_path;
            {
                auto output_dir = tmp_dir / "Mesh";
                std::filesystem::create_directories(output_dir);
                output_mesh_path = output_dir / "Temp.obj";
                python_system_.SetTupleItem(*args, 1, python_system_.MakeObject(output_mesh_path.wstring()));
            }

            std::cout << "Generating mesh by AI...\n";
            python_system_.CallObject(*mesh_generator_gen_method_, *args);

            return LoadMesh(output_mesh_path);
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

    MeshGenerator::MeshGenerator(PythonSystem& python_system) : impl_(std::make_unique<Impl>(python_system))
    {
    }

    MeshGenerator::~MeshGenerator() noexcept = default;

    MeshGenerator::MeshGenerator(MeshGenerator&& other) noexcept = default;
    MeshGenerator& MeshGenerator::operator=(MeshGenerator&& other) noexcept = default;

    Mesh MeshGenerator::Generate(std::span<const Texture> input_images, const std::filesystem::path& tmp_dir)
    {
        return impl_->Generate(std::move(input_images), tmp_dir);
    }
} // namespace AIHoloImager
