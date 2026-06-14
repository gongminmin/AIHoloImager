// Copyright (c) 2024-2026 Minmin Gong
//

#pragma once

#include <memory>

#include "AIHoloImager/Mesh.hpp"
#include "Base/Noncopyable.hpp"

namespace AIHoloImager
{
    class MeshSimplification
    {
        DISALLOW_COPY_AND_ASSIGN(MeshSimplification);

    public:
        MeshSimplification();
        MeshSimplification(MeshSimplification&& other) noexcept;
        ~MeshSimplification() noexcept;

        MeshSimplification& operator=(MeshSimplification&& other) noexcept;

        // The input mesh can have any vertex attributes, but the simplified mesh has only the position left.
        Mesh Process(const Mesh& mesh, float face_ratio);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
