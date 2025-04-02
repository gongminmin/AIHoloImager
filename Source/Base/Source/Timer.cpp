// Copyright (c) 2024-2025 Minmin Gong
//

#include "Base/Timer.hpp"

namespace AIHoloImager
{
    Timer::Timer() noexcept
    {
        this->Restart();
    }

    void Timer::Restart() noexcept
    {
        start_time_ = this->Now();
    }

    std::chrono::duration<double> Timer::Elapsed() const noexcept
    {
        return std::chrono::duration<double>(this->Now() - start_time_);
    }

    std::chrono::high_resolution_clock::time_point Timer::Now() const noexcept
    {
        return std::chrono::high_resolution_clock::now();
    }
} // namespace AIHoloImager
