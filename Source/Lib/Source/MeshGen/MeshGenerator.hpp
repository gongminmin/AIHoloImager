// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include "AIHoloImager/Mesh.hpp"
#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"
#include "SfM/StructureFromMotion.hpp"

namespace AIHoloImager
{
    class MeshGenerator
    {
        DISALLOW_COPY_AND_ASSIGN(MeshGenerator);

    public:
        explicit MeshGenerator(AIHoloImagerInternal& aihi);
        MeshGenerator(MeshGenerator&& other) noexcept;
        ~MeshGenerator() noexcept;

        MeshGenerator& operator=(MeshGenerator&& other) noexcept;

        Mesh Generate(const StructureFromMotion::Result& sfm_input, uint32_t texture_size);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
