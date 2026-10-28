// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <memory>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuBuffer.hpp"
#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuSystem.hpp"
#include "GpuComputationLib/Symbol.hpp"

namespace AIHoloImager
{
    class Sorter
    {
        DISALLOW_COPY_AND_ASSIGN(Sorter);

    public:
        AIHI_GPU_COMPUTATION_LIB_API Sorter() noexcept;
        AIHI_GPU_COMPUTATION_LIB_API explicit Sorter(GpuSystem& gpu_system);
        AIHI_GPU_COMPUTATION_LIB_API Sorter(Sorter&& other) noexcept;
        AIHI_GPU_COMPUTATION_LIB_API ~Sorter() noexcept;

        AIHI_GPU_COMPUTATION_LIB_API Sorter& operator=(Sorter&& other) noexcept;

        AIHI_GPU_COMPUTATION_LIB_API void RadixSort(GpuCommandList& cmd_list, const GpuBuffer& keys, GpuFormat key_format,
            const GpuBuffer& values, GpuFormat value_format, uint32_t num_items, GpuBuffer& sorted_keys, GpuBuffer& sorted_values,
            uint32_t bits);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
