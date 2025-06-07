// Copyright (c) 2025 Minmin Gong
//

#include "PerfProfiler.hpp"

#include <cassert>
#include <format>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>

#include "Base/Timer.hpp"

namespace AIHoloImager
{
    class PerfProfiler::Impl final
    {
    public:
        Timer& CpuTimer() noexcept
        {
            return cpu_timer_;
        }

        void EnterRegion(std::string_view name)
        {
            std::lock_guard lock(perf_mutex_);

            auto thread_id = std::this_thread::get_id();
            auto iter = perf_data_.find(thread_id);
            if (iter == perf_data_.end())
            {
                iter = perf_data_.emplace(std::move(thread_id), ThreadPerfInfo{}).first;
            }
            else
            {
                ++iter->second.level;
            }
            iter->second.regions.emplace_back(iter->second.level, std::string(std::move(name)), std::chrono::milliseconds{});
        }

        void LeaveRegion(std::string_view name, std::chrono::milliseconds duration)
        {
            std::lock_guard lock(perf_mutex_);

            auto thread_id = std::this_thread::get_id();
            auto iter = perf_data_.find(thread_id);
            assert(iter != perf_data_.end());

            for (auto region_iter = iter->second.regions.rbegin(); region_iter != iter->second.regions.rend(); ++region_iter)
            {
                if ((region_iter->name == name) && (region_iter->level == iter->second.level))
                {
                    region_iter->duration = std::move(duration);
                    break;
                }
            }

            --iter->second.level;
        }

        void Output(std::ostream& os) const
        {
            std::lock_guard lock(perf_mutex_);

            os << "\nTimings:\n";
            os << "=========================\n";
            for (auto&& [thread_id, perf_info] : perf_data_)
            {
                os << "Thread: " << thread_id << '\n';
                for (const auto& region : perf_info.regions)
                {
                    for (uint32_t i = 0; i <= region.level; ++i)
                    {
                        os << "  ";
                    }

                    os << std::format("{}: ", region.name);
                    if (region.duration > std::chrono::seconds(1))
                    {
                        os << std::format("{:.3f} s", std::chrono::duration_cast<std::chrono::duration<float>>(region.duration).count());
                    }
                    else
                    {
                        os << std::format("{} ms", region.duration.count());
                    }
                    os << '\n';
                }
                os << "=========================\n";
            }
            os << '\n';
        }

    private:
        Timer cpu_timer_;

        struct PerfInfo
        {
            uint32_t level = 0;
            std::string name;
            std::chrono::milliseconds duration;
        };
        struct ThreadPerfInfo
        {
            uint32_t level = 0;
            std::vector<PerfInfo> regions;
        };
        std::map<std::thread::id, ThreadPerfInfo> perf_data_;
        mutable std::mutex perf_mutex_;
    };

    PerfProfiler::PerfProfiler() : impl_(std::make_unique<Impl>())
    {
    }

    PerfProfiler::~PerfProfiler() noexcept = default;

    void PerfProfiler::Output(std::ostream& os) const
    {
        impl_->Output(os);
    }

    class PerfRegion::Impl final
    {
    public:
        Impl(PerfProfiler& profiler, std::string_view name) noexcept : profiler_(profiler), name_(std::move(name))
        {
            profiler_.impl_->EnterRegion(name_);
            start_time_ = profiler_.impl_->CpuTimer().Now();
        }

        ~Impl()
        {
            const auto end_time = profiler_.impl_->CpuTimer().Now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_);
            profiler_.impl_->LeaveRegion(name_, std::move(duration));
        }

    private:
        PerfProfiler& profiler_;
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_time_;
    };

    PerfRegion::PerfRegion(PerfProfiler& profiler, std::string_view name) : impl_(std::make_unique<Impl>(profiler, std::move(name)))
    {
    }

    PerfRegion::~PerfRegion() = default;
} // namespace AIHoloImager
