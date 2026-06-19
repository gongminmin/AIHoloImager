// Copyright (c) 2024-2026 Minmin Gong
//

#pragma once

#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"

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

        void Generate(AIHoloImagerInternal::ProjectionDesc& projection);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
