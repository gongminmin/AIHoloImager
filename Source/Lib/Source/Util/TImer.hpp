// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <chrono>

namespace AIHoloImager
{
    class Timer final
    {
    public:
        Timer() noexcept;

        void Restart() noexcept;
        std::chrono::high_resolution_clock::time_point Now() const noexcept;

        std::chrono::duration<double> Elapsed() const noexcept;

        template <typename T>
        T ElapsedOfType() const noexcept
        {
            return std::chrono::duration_cast<T>(this->Elapsed());
        }

    private:
        std::chrono::high_resolution_clock::time_point start_time_;
    };
} // namespace AIHoloImager