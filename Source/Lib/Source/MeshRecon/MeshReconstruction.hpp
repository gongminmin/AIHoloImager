// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <filesystem>
#include <memory>

#include "SfM/StructureFromMotion.hpp"

namespace AIHoloImager
{
    class MeshReconstruction
    {
    public:
        explicit MeshReconstruction(const std::filesystem::path& exe_path);
        MeshReconstruction(const MeshReconstruction& other) = delete;
        MeshReconstruction(MeshReconstruction&& other) noexcept;
        ~MeshReconstruction() noexcept;

        MeshReconstruction& operator=(const MeshReconstruction& other) = delete;
        MeshReconstruction& operator=(MeshReconstruction&& other) noexcept;

        void Process(const StructureFromMotion::Result& sfm_input, const std::filesystem::path& tmp_dir);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
