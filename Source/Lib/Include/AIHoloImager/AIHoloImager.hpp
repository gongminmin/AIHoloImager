// Copyright (c) 2024 Minmin Gong
//

#include <filesystem>
#include <memory>

#include "AIHoloImager/Mesh.hpp"

namespace AIHoloImager
{
    class AIHoloImager
    {
    public:
        explicit AIHoloImager(const std::filesystem::path& tmp_dir);
        AIHoloImager(const AIHoloImager& rhs) = delete;
        AIHoloImager(AIHoloImager&& rhs) noexcept;
        ~AIHoloImager() noexcept;

        AIHoloImager& operator=(const AIHoloImager& rhs) = delete;
        AIHoloImager& operator=(AIHoloImager&& rhs) noexcept;

        Mesh Generate(const std::filesystem::path& input_path);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
