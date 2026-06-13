// Copyright (c) 2025-2026 Minmin Gong
//

#pragma once

#include <memory>

#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"
#include "Util/GpuMesh.hpp"

namespace AIHoloImager
{
    class InvisibleFacesRemover
    {
        DISALLOW_COPY_AND_ASSIGN(InvisibleFacesRemover);

    public:
        explicit InvisibleFacesRemover(AIHoloImagerInternal& aihi);
        InvisibleFacesRemover(InvisibleFacesRemover&& other) noexcept;
        ~InvisibleFacesRemover() noexcept;

        InvisibleFacesRemover& operator=(InvisibleFacesRemover&& other) noexcept;

        GpuMesh Process(const GpuMesh& mesh);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
