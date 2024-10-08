// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <filesystem>
#include <memory>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif
#include <DirectXCollision.h>
#include <DirectXMath.h>

#include "AIHoloImager/Mesh.hpp"
#include "Gpu/GpuSystem.hpp"
#include "SfM/StructureFromMotion.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class MeshReconstruction
    {
        DISALLOW_COPY_AND_ASSIGN(MeshReconstruction);

    public:
        struct Result
        {
            Mesh mesh;

            DirectX::XMFLOAT4X4 transform; // From model space to SfM space
            DirectX::BoundingOrientedBox obb;
        };

    public:
        MeshReconstruction(const std::filesystem::path& exe_dir, GpuSystem& gpu_system);
        MeshReconstruction(MeshReconstruction&& other) noexcept;
        ~MeshReconstruction() noexcept;

        MeshReconstruction& operator=(MeshReconstruction&& other) noexcept;

        Result Process(const StructureFromMotion::Result& sfm_input, bool refine_mesh, uint32_t max_texture_size,
            const std::filesystem::path& tmp_dir);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
