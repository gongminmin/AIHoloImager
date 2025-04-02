// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <memory>

#include "AIHoloImager/Mesh.hpp"
#include "Base/Noncopyable.hpp"
#include "Gpu/GpuSystem.hpp"

namespace AIHoloImager
{
    class InvisibleFacesRemover
    {
        DISALLOW_COPY_AND_ASSIGN(InvisibleFacesRemover);

    public:
        explicit InvisibleFacesRemover(GpuSystem& gpu_system);
        InvisibleFacesRemover(InvisibleFacesRemover&& other) noexcept;
        ~InvisibleFacesRemover() noexcept;

        InvisibleFacesRemover& operator=(InvisibleFacesRemover&& other) noexcept;

        Mesh Process(const Mesh& mesh);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
