// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "AIHoloImager/Texture.hpp"
#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"

namespace AIHoloImager
{
    class Delighter
    {
        DISALLOW_COPY_AND_ASSIGN(Delighter);

    public:
        explicit Delighter(AIHoloImagerInternal& aihi);
        Delighter(Delighter&& other) noexcept;
        ~Delighter() noexcept;

        Delighter& operator=(Delighter&& other) noexcept;

        Texture Process(const Texture& image, const glm::uvec4& roi, glm::uvec2& offset);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
