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
#include "Gpu/GpuQuery.hpp"

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

        void LeaveRegion(std::string_view name, std::chrono::milliseconds duration, GpuTimerQuery gpu_timer_query)
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
                    region_iter->gpu_timer_query = std::move(gpu_timer_query);
                    break;
                }
            }

            --iter->second.level;
        }

        void Output(std::ostream& os) const
        {
            std::lock_guard lock(perf_mutex_);

            auto format_duration = [](std::chrono::milliseconds duration) {
                if (duration > std::chrono::seconds(1))
                {
                    return std::format("{:.3f} s", std::chrono::duration_cast<std::chrono::duration<float>>(duration).count());
                }
                else
                {
                    return std::format("{} ms", duration.count());
                }
            };

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

                    os << "CPU " << format_duration(region.duration);
                    if (region.gpu_timer_query)
                    {
                        const auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(region.gpu_timer_query.Elapsed());
                        os << "  GPU " << format_duration(gpu_duration);
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
            GpuTimerQuery gpu_timer_query;
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
        Impl(PerfProfiler& profiler, std::string_view name, GpuCommandList* gpu_cmd_list) noexcept
            : profiler_(profiler), name_(std::move(name)), gpu_cmd_list_(gpu_cmd_list)
        {
            profiler_.impl_->EnterRegion(name_);
            start_time_ = profiler_.impl_->CpuTimer().Now();
            if (gpu_cmd_list_ != nullptr)
            {
                gpu_timer_query_ = GpuTimerQuery(gpu_cmd_list->GpuSys());
                gpu_timer_query_.Begin(*gpu_cmd_list_);
            }
        }

        ~Impl()
        {
            const auto end_time = profiler_.impl_->CpuTimer().Now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_);
            if (gpu_cmd_list_ != nullptr)
            {
                gpu_timer_query_.End(*gpu_cmd_list_);
            }
            profiler_.impl_->LeaveRegion(name_, std::move(duration), std::move(gpu_timer_query_));
        }

    private:
        PerfProfiler& profiler_;
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_time_;
        GpuCommandList* gpu_cmd_list_;
        GpuTimerQuery gpu_timer_query_;
    };

    PerfRegion::PerfRegion(PerfProfiler& profiler, std::string_view name, GpuCommandList* gpu_cmd_list)
        : impl_(std::make_unique<Impl>(profiler, std::move(name), gpu_cmd_list))
    {
    }

    PerfRegion::~PerfRegion() = default;
} // namespace AIHoloImager
