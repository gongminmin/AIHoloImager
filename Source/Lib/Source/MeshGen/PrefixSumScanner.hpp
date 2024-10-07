// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include "Gpu/GpuBuffer.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class PrefixSumScanner
    {
        DISALLOW_COPY_AND_ASSIGN(PrefixSumScanner);

    public:
        explicit PrefixSumScanner(GpuSystem& gpu_system);
        PrefixSumScanner(PrefixSumScanner&& other) noexcept;
        ~PrefixSumScanner() noexcept;

        PrefixSumScanner& operator=(PrefixSumScanner&& other) noexcept;

        void Scan(GpuCommandList& cmd_list, const GpuBuffer& input, GpuBuffer& output, uint32_t num_elems, DXGI_FORMAT format);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
