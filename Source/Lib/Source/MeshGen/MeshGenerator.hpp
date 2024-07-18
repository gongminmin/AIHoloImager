// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <filesystem>
#include <span>

#include "AIHoloImager/Mesh.hpp"
#include "AIHoloImager/Texture.hpp"
#include "Python/PythonSystem.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class MeshGenerator
    {
        DISALLOW_COPY_AND_ASSIGN(MeshGenerator);

    public:
        MeshGenerator(const std::filesystem::path& exe_dir, PythonSystem& python_system);
        MeshGenerator(MeshGenerator&& other) noexcept;
        ~MeshGenerator() noexcept;

        MeshGenerator& operator=(MeshGenerator&& other) noexcept;

        Mesh Generate(std::span<const Texture> input_images, uint32_t texture_size, const std::filesystem::path& tmp_dir);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
