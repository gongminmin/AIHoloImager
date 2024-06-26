// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <filesystem>
#include <memory>

#include "SfM/StructureFromMotion.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class MeshReconstruction
    {
        DISALLOW_COPY_AND_ASSIGN(MeshReconstruction);

    public:
        explicit MeshReconstruction(const std::filesystem::path& exe_dir);
        MeshReconstruction(MeshReconstruction&& other) noexcept;
        ~MeshReconstruction() noexcept;

        MeshReconstruction& operator=(MeshReconstruction&& other) noexcept;

        void Process(const StructureFromMotion::Result& sfm_input, bool refine_mesh, uint32_t max_texture_size,
            const std::filesystem::path& tmp_dir);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
