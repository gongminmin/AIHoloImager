// Copyright (c) 2024-2026 Minmin Gong
//

#pragma once

#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"
#include "Gpu/GpuBuffer.hpp"
#include "Gpu/GpuFormat.hpp"

namespace AIHoloImager
{
    class PrefixSumScanner
    {
        DISALLOW_COPY_AND_ASSIGN(PrefixSumScanner);

    public:
        PrefixSumScanner() noexcept;
        explicit PrefixSumScanner(AIHoloImagerInternal& aihi);
        PrefixSumScanner(PrefixSumScanner&& other) noexcept;
        ~PrefixSumScanner() noexcept;

        PrefixSumScanner& operator=(PrefixSumScanner&& other) noexcept;

        void Scan(GpuCommandList& cmd_list, const GpuBuffer& input, GpuBuffer& output, uint32_t num_elems, GpuFormat format,
            bool exclusive = true);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
