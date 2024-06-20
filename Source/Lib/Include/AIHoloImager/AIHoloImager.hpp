// Copyright (c) 2024 Minmin Gong
//

#include <memory>
#include <span>

#include "AIHoloImager/Mesh.hpp"
#include "AIHoloImager/Texture.hpp"

namespace AIHoloImager
{
    class AIHoloImager
    {
    public:
        AIHoloImager();
        AIHoloImager(const AIHoloImager& rhs) = delete;
        AIHoloImager(AIHoloImager&& rhs) noexcept;
        ~AIHoloImager() noexcept;

        AIHoloImager& operator=(const AIHoloImager& rhs) = delete;
        AIHoloImager& operator=(AIHoloImager&& rhs) noexcept;

        Mesh Generate(std::span<const Texture> images);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
