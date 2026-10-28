// Copyright (c) 2024-2026 Minmin Gong
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
    class PrefixSumScanner
    {
        DISALLOW_COPY_AND_ASSIGN(PrefixSumScanner);

    public:
        AIHI_GPU_COMPUTATION_LIB_API PrefixSumScanner() noexcept;
        AIHI_GPU_COMPUTATION_LIB_API explicit PrefixSumScanner(GpuSystem& gpu_system);
        AIHI_GPU_COMPUTATION_LIB_API PrefixSumScanner(PrefixSumScanner&& other) noexcept;
        AIHI_GPU_COMPUTATION_LIB_API ~PrefixSumScanner() noexcept;

        AIHI_GPU_COMPUTATION_LIB_API PrefixSumScanner& operator=(PrefixSumScanner&& other) noexcept;

        AIHI_GPU_COMPUTATION_LIB_API void Scan(GpuCommandList& cmd_list, const GpuBuffer& input, GpuBuffer& output, uint32_t num_elems,
            GpuFormat format, bool exclusive = true);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
