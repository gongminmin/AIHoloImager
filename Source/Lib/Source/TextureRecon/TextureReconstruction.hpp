// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <glm/mat4x4.hpp>

#include "AIHoloImager/Mesh.hpp"
#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"
#include "Gpu/GpuTexture.hpp"

namespace AIHoloImager
{
    class TextureReconstruction
    {
        DISALLOW_COPY_AND_ASSIGN(TextureReconstruction);

    public:
        struct ProjectionDesc
        {
            const GpuTexture2D* image;

            glm::mat4x4 view_mtx;
            glm::mat4x4 proj_mtx;
            uint32_t full_width;
            uint32_t full_height;
            glm::vec2 vp_offset;
            glm::uvec2 image_offset;
        };

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

        Result Process(const Mesh& mesh, const glm::mat4x4& model_mtx, std::span<const ProjectionDesc> projections, uint32_t texture_size);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
