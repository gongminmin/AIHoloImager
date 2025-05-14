// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <glm/vec4.hpp>

#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuTexture.hpp"

namespace AIHoloImager
{
    class MaskGenerator
    {
        DISALLOW_COPY_AND_ASSIGN(MaskGenerator);

    public:
        explicit MaskGenerator(AIHoloImagerInternal& aihi);
        MaskGenerator(MaskGenerator&& other) noexcept;
        ~MaskGenerator() noexcept;

        MaskGenerator& operator=(MaskGenerator&& other) noexcept;

        void Generate(GpuCommandList& cmd_list, GpuTexture2D& image, glm::uvec4& roi);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
