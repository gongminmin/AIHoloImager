// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <filesystem>
#include <memory>

#include "AIHoloImager/Mesh.hpp"

namespace AIHoloImager
{
    class AIHoloImager
    {
    public:
        enum class DeviceType
        {
            Cpu,
            Cuda,
        };
        enum class Api
        {
            D3D12,
            Vulkan,

            Auto,
        };

    public:
        AIHoloImager(DeviceType device, Api api, const std::filesystem::path& tmp_dir, bool gpu_debug = false);
        AIHoloImager(const AIHoloImager& rhs) = delete;
        AIHoloImager(AIHoloImager&& rhs) noexcept;
        ~AIHoloImager() noexcept;

        AIHoloImager& operator=(const AIHoloImager& rhs) = delete;
        AIHoloImager& operator=(AIHoloImager&& rhs) noexcept;

        Mesh Generate(const std::filesystem::path& input_path, uint32_t texture_size, bool no_delight = false);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
