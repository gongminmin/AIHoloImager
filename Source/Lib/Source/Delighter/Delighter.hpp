// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <glm/vec4.hpp>

#include "AIHoloImager/Texture.hpp"
#include "Python/PythonSystem.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class Delighter
    {
        DISALLOW_COPY_AND_ASSIGN(Delighter);

    public:
        explicit Delighter(PythonSystem& python_system);
        Delighter(Delighter&& other) noexcept;
        ~Delighter() noexcept;

        Delighter& operator=(Delighter&& other) noexcept;

        void ProcessInPlace(Texture& inout_image, const glm::uvec4& roi);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
