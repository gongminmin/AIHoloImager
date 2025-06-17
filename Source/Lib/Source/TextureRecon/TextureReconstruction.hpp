// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <filesystem>

#include <glm/mat4x4.hpp>

#include "AIHoloImager/Mesh.hpp"
#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"
#include "Gpu/GpuTexture.hpp"
#include "SfM/StructureFromMotion.hpp"
#include "Util/BoundingBox.hpp"

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
        };

    public:
        explicit TextureReconstruction(AIHoloImagerInternal& aihi);
        TextureReconstruction(TextureReconstruction&& other) noexcept;
        ~TextureReconstruction() noexcept;

        TextureReconstruction& operator=(TextureReconstruction&& other) noexcept;

        Result Process(const Mesh& mesh, const glm::mat4x4& model_mtx, const Obb& world_obb, const StructureFromMotion::Result& sfm_input,
            uint32_t texture_size);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
