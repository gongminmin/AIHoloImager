// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <memory>
#include <span>

#include "AIHoloImager/Texture.hpp"

namespace AIHoloImager
{
    class StructureFromMotion
    {
    public:
        StructureFromMotion();
        StructureFromMotion(const StructureFromMotion& other) = delete;
        StructureFromMotion(StructureFromMotion&& other) noexcept;
        ~StructureFromMotion() noexcept;

        StructureFromMotion& operator=(const StructureFromMotion& other) = delete;
        StructureFromMotion& operator=(StructureFromMotion&& other) noexcept;

        void Process(const std::filesystem::path& input_path);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
