// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <memory>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuSampler.hpp"
#include "Gpu/GpuTexture.hpp"

namespace AIHoloImager
{
    class GpuSystem;
    class GpuCommandList;

    class GpuMipmapper final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuMipmapper)

    public:
        GpuMipmapper();
        explicit GpuMipmapper(GpuSystem& gpu_system);
        ~GpuMipmapper() noexcept;

        GpuMipmapper(GpuMipmapper&& other) noexcept;
        GpuMipmapper& operator=(GpuMipmapper&& other) noexcept;

        void Generate(GpuCommandList& cmd_list, GpuTexture2D& texture, GpuSampler::Filter filter);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
