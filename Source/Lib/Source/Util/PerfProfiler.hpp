// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <memory>
#include <string_view>

#include "Base/NonCopyable.hpp"

namespace AIHoloImager
{
    class PerfRegion;

    class PerfProfiler final
    {
        friend class PerfRegion;

        DISALLOW_COPY_AND_ASSIGN(PerfProfiler);

    public:
        PerfProfiler();
        ~PerfProfiler() noexcept;

        void Output(std::ostream& os) const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class PerfRegion final
    {
        DISALLOW_COPY_AND_ASSIGN(PerfRegion);

    public:
        PerfRegion(PerfProfiler& profiler, std::string_view name);
        ~PerfRegion();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
