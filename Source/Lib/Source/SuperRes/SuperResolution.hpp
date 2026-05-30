// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"

namespace AIHoloImager
{
    class SuperResolution final
    {
        DISALLOW_COPY_AND_ASSIGN(SuperResolution);

    public:
        SuperResolution() noexcept;
        explicit SuperResolution(AIHoloImagerInternal& aihi);
        SuperResolution(SuperResolution&& other) noexcept;
        ~SuperResolution() noexcept;

        SuperResolution& operator=(SuperResolution&& other) noexcept;

        AIHoloImagerInternal::ProjectionDesc Process(const AIHoloImagerInternal::ProjectionDesc& original, float scale);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
