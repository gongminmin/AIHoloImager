// Copyright (c) 2025-2026 Minmin Gong
//

#include "GpuSystemInternal.hpp"

#include <cassert>

namespace AIHoloImager
{
    GpuSystemInternal::GpuSystemInternal() noexcept = default;
    GpuSystemInternal::GpuSystemInternal(GpuSystem& gpu_system, bool enable_sharing)
        : gpu_system_(&gpu_system), enable_sharing_(enable_sharing)
    {
    }
    GpuSystemInternal::~GpuSystemInternal() = default;

    GpuSystemInternal::GpuSystemInternal(GpuSystemInternal&& other) noexcept = default;
    GpuSystemInternal& GpuSystemInternal::operator=(GpuSystemInternal&& other) noexcept = default;

    GpuSystem& GpuSystemInternal::GpuSys() const noexcept
    {
        return *gpu_system_;
    }

    GpuCommandQueue* GpuSystemInternal::CommandQueue(GpuSystem::CmdQueueType type) noexcept
    {
        auto* cmd_queue_ctx = this->GetCommandQueueContext(type);
        return cmd_queue_ctx ? &cmd_queue_ctx->cmd_queue : nullptr;
    }

    void* GpuSystemInternal::SharedFenceHandle(GpuSystem::CmdQueueType type) const noexcept
    {
        auto* cmd_queue_ctx = this->GetCommandQueueContext(type);
        return cmd_queue_ctx ? cmd_queue_ctx->fence.SharedFenceHandle() : nullptr;
    }

    GpuCommandList GpuSystemInternal::CreateCommandList(GpuSystem::CmdQueueType type, std::string_view name)
    {
        GpuCommandList cmd_list;
        auto& cmd_pool = this->CurrentCommandPool(type);
        auto& cmd_queue_ctx = *this->GetCommandQueueContext(type);
        if (cmd_queue_ctx.free_cmd_lists.empty())
        {
            cmd_list = GpuCommandList(*gpu_system_, cmd_pool, type);
        }
        else
        {
            cmd_list = std::move(cmd_queue_ctx.free_cmd_lists.front());
            cmd_queue_ctx.free_cmd_lists.pop_front();
            cmd_list.Reset(cmd_pool);
        }
        cmd_list.Name(std::move(name));
        return cmd_list;
    }

    uint64_t GpuSystemInternal::Execute(GpuCommandList&& cmd_list, const GpuSystem::WaitFences& wait_fences)
    {
        const uint64_t new_fence_value = this->ExecuteOnly(cmd_list.Internal(), wait_fences);
        this->GetCommandQueueContext(cmd_list.Type())->free_cmd_lists.emplace_back(std::move(cmd_list));
        return new_fence_value;
    }

    uint64_t GpuSystemInternal::ExecuteAndReset(GpuCommandList& cmd_list, const GpuSystem::WaitFences& wait_fences)
    {
        return this->ExecuteAndReset(cmd_list.Internal(), wait_fences);
    }
    uint64_t GpuSystemInternal::ExecuteAndReset(GpuCommandListInternal& cmd_list_internal, const GpuSystem::WaitFences& wait_fences)
    {
        const uint64_t new_fence_value = this->ExecuteOnly(cmd_list_internal, wait_fences);
        cmd_list_internal.Reset(this->CurrentCommandPool(cmd_list_internal.Type()));
        return new_fence_value;
    }

    void GpuSystemInternal::CpuWait(const GpuSystem::WaitFences& wait_fences)
    {
        for (size_t i = 0; i < std::size(wait_fences.fence_values); ++i)
        {
            const auto queue_type = static_cast<GpuSystem::CmdQueueType>(i);
            auto* wait_cmd_queue_ctx = this->GetCommandQueueContext(queue_type);
            if ((wait_cmd_queue_ctx != nullptr) && (wait_cmd_queue_ctx->fence_val != 0) && (wait_fences.fence_values[i] != 0))
            {
                const uint64_t wait_fence_value = wait_fences.fence_values[i] == GpuSystem::MaxFenceValue
                                                      ? wait_cmd_queue_ctx->fence_val - 1
                                                      : wait_fences.fence_values[i];
                if (wait_cmd_queue_ctx->fence.CompletedValue() < wait_fence_value)
                {
                    wait_cmd_queue_ctx->fence.CpuWait(wait_fence_value);
                }
            }
        }

        this->ClearStallResources();
    }

    void GpuSystemInternal::GpuWait(GpuSystem::CmdQueueType target_queue_type, const GpuSystem::WaitFences& wait_fences)
    {
        auto* target_cmd_queue_ctx = this->GetCommandQueueContext(target_queue_type);
        if (target_cmd_queue_ctx != nullptr)
        {
            GpuCommandQueue::FenceInfo wait_fence_values[static_cast<uint32_t>(GpuSystem::CmdQueueType::Num)];
            uint32_t num_waits = 0;
            for (size_t i = 0; i < std::size(wait_fences.fence_values); ++i)
            {
                const auto queue_type = static_cast<GpuSystem::CmdQueueType>(i);
                auto* wait_cmd_queue_ctx = this->GetCommandQueueContext(queue_type);
                if ((wait_cmd_queue_ctx != nullptr) && (wait_cmd_queue_ctx->fence_val != 0) && (wait_fences.fence_values[i] != 0))
                {
                    const uint64_t wait_value = wait_fences.fence_values[i] == GpuSystem::MaxFenceValue ? wait_cmd_queue_ctx->fence_val - 1
                                                                                                        : wait_fences.fence_values[i];
                    wait_fence_values[num_waits] = {&wait_cmd_queue_ctx->fence, wait_value};

                    if (target_queue_type == queue_type)
                    {
                        target_cmd_queue_ctx->fence_val = std::max(target_cmd_queue_ctx->fence_val, wait_value) + 1;
                    }

                    ++num_waits;
                }
            }

            target_cmd_queue_ctx->cmd_queue.GpuWait(std::span(wait_fence_values, num_waits));
        }
    }

    uint64_t GpuSystemInternal::FenceValue(GpuSystem::CmdQueueType type) const noexcept
    {
        auto* cmd_queue_ctx = this->GetCommandQueueContext(type);
        return cmd_queue_ctx ? cmd_queue_ctx->fence_val : 0;
    }

    uint64_t GpuSystemInternal::CompletedFenceValue(GpuSystem::CmdQueueType type) const
    {
        auto* cmd_queue_ctx = this->GetCommandQueueContext(type);
        return cmd_queue_ctx ? cmd_queue_ctx->fence.CompletedValue() : 0;
    }

    GpuSystemInternal::CommandQueueContext& GpuSystemInternal::GetOrCreateCommandQueueContext(GpuSystem::CmdQueueType type)
    {
        auto& cmd_queue_ctx = cmd_queue_ctxs_[static_cast<uint32_t>(type)];
        if (!cmd_queue_ctx.cmd_queue)
        {
            const std::string debug_name = std::format("cmd_queue {}", static_cast<uint32_t>(type));
            cmd_queue_ctx.cmd_queue = GpuCommandQueue(*gpu_system_, type, debug_name);

            cmd_queue_ctx.fence = GpuFence(*gpu_system_, cmd_queue_ctx.fence_val, enable_sharing_);
            ++cmd_queue_ctx.fence_val;
        }

        return cmd_queue_ctx;
    }

    GpuSystemInternal::CommandQueueContext* GpuSystemInternal::GetCommandQueueContext(GpuSystem::CmdQueueType type)
    {
        auto& cmd_queue_ctx = cmd_queue_ctxs_[static_cast<uint32_t>(type)];
        if (cmd_queue_ctx.cmd_queue)
        {
            return &cmd_queue_ctx;
        }
        return nullptr;
    }
    const GpuSystemInternal::CommandQueueContext* GpuSystemInternal::GetCommandQueueContext(GpuSystem::CmdQueueType type) const
    {
        return const_cast<GpuSystemInternal*>(this)->GetCommandQueueContext(type);
    }

    void GpuSystemInternal::ClearCommandQueueContexts()
    {
        for (auto& cmd_queue_ctx : cmd_queue_ctxs_)
        {
            cmd_queue_ctx.cmd_queue = {};
            cmd_queue_ctx.cmd_pools.clear();
            cmd_queue_ctx.free_cmd_lists.clear();

            cmd_queue_ctx.fence = {};
        }
    }

    GpuCommandPool& GpuSystemInternal::CurrentCommandPool(GpuSystem::CmdQueueType type)
    {
        auto& cmd_queue_ctx = this->GetOrCreateCommandQueueContext(type);
        const uint64_t completed_fence = cmd_queue_ctx.fence.CompletedValue();
        for (auto& pool : cmd_queue_ctx.cmd_pools)
        {
            auto& pool_internal = pool->Internal();
            if (pool_internal.Empty() && (pool_internal.FenceValue() <= completed_fence))
            {
                pool_internal.Reset();
                return *pool;
            }
        }

        return *cmd_queue_ctx.cmd_pools.emplace_back(std::make_unique<GpuCommandPool>(*gpu_system_, type));
    }

    uint64_t GpuSystemInternal::ExecuteOnly(GpuCommandListInternal& cmd_list_internal, const GpuSystem::WaitFences& wait_fences)
    {
        auto& cmd_pool_internal = cmd_list_internal.CommandPool().Internal();
        cmd_list_internal.Close();

        auto* cmd_queue_ctx = this->GetCommandQueueContext(cmd_list_internal.Type());
        assert(cmd_queue_ctx != nullptr);

        GpuSystem::WaitFences dep_wait_fences;
        cmd_list_internal.WaitForFences(dep_wait_fences);

        GpuCommandQueue::FenceInfo compact_wait_fence_values[static_cast<uint32_t>(GpuSystem::CmdQueueType::Num)];
        uint32_t num_waits = 0;
        for (size_t i = 0; i < std::size(dep_wait_fences.fence_values); ++i)
        {
            auto* wait_cmd_queue_ctx = this->GetCommandQueueContext(static_cast<GpuSystem::CmdQueueType>(i));
            if ((wait_cmd_queue_ctx != nullptr) && (wait_cmd_queue_ctx->fence_val != 0))
            {
                uint64_t fence_value = dep_wait_fences.fence_values[i];
                if (wait_fences.fence_values[i] != 0)
                {
                    fence_value =
                        std::max(fence_value, wait_fences.fence_values[i] == GpuSystem::MaxFenceValue ? wait_cmd_queue_ctx->fence_val - 1
                                                                                                      : wait_fences.fence_values[i]);
                }
                if (fence_value != 0)
                {
                    compact_wait_fence_values[num_waits] = {&wait_cmd_queue_ctx->fence, fence_value};
                    ++num_waits;
                }
            }
        }

        const uint64_t curr_fence_value = cmd_queue_ctx->fence_val;
        cmd_list_internal.UpdateAccessInfo(curr_fence_value);
        ++cmd_queue_ctx->fence_val;

        cmd_queue_ctx->cmd_queue.Internal().Execute(
            cmd_list_internal, std::span(compact_wait_fence_values, num_waits), {&cmd_queue_ctx->fence, curr_fence_value});

        cmd_pool_internal.FenceValue(cmd_queue_ctx->fence_val);

        this->ClearStallResources();

        return curr_fence_value;
    }
} // namespace AIHoloImager
