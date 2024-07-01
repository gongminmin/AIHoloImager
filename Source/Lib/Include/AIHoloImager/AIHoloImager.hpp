// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <filesystem>
#include <memory>

#include "AIHoloImager/Common.hpp"
#include "AIHoloImager/Mesh.hpp"

namespace AIHoloImager
{
    class AIHoloImager
    {
    public:
        AIHI_API explicit AIHoloImager(const std::filesystem::path& tmp_dir);
        AIHoloImager(const AIHoloImager& rhs) = delete;
        AIHI_API AIHoloImager(AIHoloImager&& rhs) noexcept;
        AIHI_API ~AIHoloImager() noexcept;

        AIHoloImager& operator=(const AIHoloImager& rhs) = delete;
        AIHI_API AIHoloImager& operator=(AIHoloImager&& rhs) noexcept;

        AIHI_API Mesh Generate(const std::filesystem::path& input_path);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
