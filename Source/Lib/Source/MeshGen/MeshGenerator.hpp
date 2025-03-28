// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <filesystem>

#include "AIHoloImager/Mesh.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Python/PythonSystem.hpp"
#include "SfM/StructureFromMotion.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class MeshGenerator
    {
        DISALLOW_COPY_AND_ASSIGN(MeshGenerator);

    public:
        MeshGenerator(GpuSystem& gpu_system, PythonSystem& python_system);
        MeshGenerator(MeshGenerator&& other) noexcept;
        ~MeshGenerator() noexcept;

        MeshGenerator& operator=(MeshGenerator&& other) noexcept;

        Mesh Generate(const StructureFromMotion::Result& sfm_input, uint32_t texture_size, const std::filesystem::path& tmp_dir);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
