// Copyright (c) 2025-2026 Minmin Gong
//

#pragma once

#include <memory>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuSampler.hpp"
#include "Gpu/Symbol.hpp"

namespace AIHoloImager
{
    class GpuSystem;
    class GpuCommandList;
    class GpuTexture2D;

    class GpuMipmapper final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuMipmapper)

    public:
        AIHI_GPU_SYS_API GpuMipmapper();
        AIHI_GPU_SYS_API explicit GpuMipmapper(GpuSystem& gpu_system);
        AIHI_GPU_SYS_API ~GpuMipmapper() noexcept;

        AIHI_GPU_SYS_API GpuMipmapper(GpuMipmapper&& other) noexcept;
        AIHI_GPU_SYS_API GpuMipmapper& operator=(GpuMipmapper&& other) noexcept;

        AIHI_GPU_SYS_API void Generate(GpuCommandList& cmd_list, GpuTexture2D& texture, GpuSampler::Filter filter);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
