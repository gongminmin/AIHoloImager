// Copyright (c) 2025-2026 Minmin Gong
//

#pragma once

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

        void Process(AIHoloImagerInternal::ProjectionDesc& projection);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
