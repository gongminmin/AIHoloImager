// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <filesystem>

#include <DirectXCollision.h>
#include <glm/mat4x4.hpp>

#include "AIHoloImager/Mesh.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuTexture.hpp"
#include "SfM/StructureFromMotion.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class TextureReconstruction
    {
        DISALLOW_COPY_AND_ASSIGN(TextureReconstruction);

    public:
        struct Result
        {
            GpuTexture2D color_tex;

            GpuTexture2D pos_tex;
            glm::mat4x4 inv_model;
        };

    public:
        TextureReconstruction(const std::filesystem::path& exe_dir, GpuSystem& gpu_system);
        TextureReconstruction(TextureReconstruction&& other) noexcept;
        ~TextureReconstruction() noexcept;

        TextureReconstruction& operator=(TextureReconstruction&& other) noexcept;

        Result Process(const Mesh& mesh, const glm::mat4x4& model_mtx, const DirectX::BoundingOrientedBox& world_obb,
            const StructureFromMotion::Result& sfm_input, uint32_t texture_size, const std::filesystem::path& tmp_dir);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
