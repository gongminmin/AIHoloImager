// Copyright (c) 2024 Minmin Gong
//

#include "AIHoloImager/AIHoloImager.hpp"

namespace AIHoloImager
{
    class AIHoloImager::Impl
    {
    public:
        Mesh Generate([[maybe_unused]] std::span<const Texture> images)
        {
            return Mesh();
        }
    };

    AIHoloImager::AIHoloImager() : impl_(std::make_unique<Impl>())
    {
    }
    AIHoloImager::AIHoloImager(AIHoloImager&& rhs) noexcept = default;
    AIHoloImager::~AIHoloImager() noexcept = default;

    AIHoloImager& AIHoloImager::operator=(AIHoloImager&& rhs) noexcept = default;

    Mesh AIHoloImager::Generate(std::span<const Texture> images)
    {
        return impl_->Generate(std::move(images));
    }
} // namespace AIHoloImager
