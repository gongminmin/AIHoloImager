// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <filesystem>
#include <memory>

#include "AIHoloImager/Mesh.hpp"
#include "MeshRecon/MeshReconstruction.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class PostProcessor
    {
        DISALLOW_COPY_AND_ASSIGN(PostProcessor);

    public:
        explicit PostProcessor(const std::filesystem::path& exe_dir);
        PostProcessor(PostProcessor&& other) noexcept;
        ~PostProcessor() noexcept;

        PostProcessor& operator=(PostProcessor&& other) noexcept;

        Mesh Process(const MeshReconstruction::Result& recon_input, const Mesh& ai_mesh, uint32_t max_texture_size,
            const std::filesystem::path& tmp_dir);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
