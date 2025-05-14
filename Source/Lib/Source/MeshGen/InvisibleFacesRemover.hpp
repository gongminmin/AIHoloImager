// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <memory>

#include "AIHoloImager/Mesh.hpp"
#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"

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

        Mesh Process(const Mesh& mesh);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
