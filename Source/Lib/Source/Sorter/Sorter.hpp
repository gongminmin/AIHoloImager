// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"
#include "Gpu/GpuBuffer.hpp"
#include "Gpu/GpuFormat.hpp"

namespace AIHoloImager
{
    class Sorter
    {
        DISALLOW_COPY_AND_ASSIGN(Sorter);

    public:
        Sorter() noexcept;
        explicit Sorter(AIHoloImagerInternal& aihi);
        Sorter(Sorter&& other) noexcept;
        ~Sorter() noexcept;

        Sorter& operator=(Sorter&& other) noexcept;

        void RadixSort(GpuCommandList& cmd_list, const GpuBuffer& keys, GpuFormat key_format, const GpuBuffer& values,
            GpuFormat value_format, uint32_t num_items, GpuBuffer& sorted_keys, GpuBuffer& sorted_values, uint32_t bits);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
