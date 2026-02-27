// Copyright (c) 2024-2026 Minmin Gong
//

#include "Gpu/GpuSystem.hpp"

#include <cassert>
#include <format>
#include <mutex>

#include "Gpu/GpuCommandList.hpp"

#include "Internal/GpuSystemInternal.hpp"
#include "Internal/GpuSystemInternalFactory.hpp"

namespace AIHoloImager
{
    class GpuSystem::Impl
    {
    public:
        Impl(Api api, GpuSystem& host, std::function<bool(Api api, void* device)> confirm_device, bool enable_sharing, bool enable_debug,
            bool enable_async_compute)
            : host_(host), api_(SelectApi(api)), enable_async_compute_(enable_async_compute),
              system_internal_(CreateGpuSystemInternal(api_, host, std::move(confirm_device), enable_sharing, enable_debug)),
              upload_mem_allocator_(host, true), read_back_mem_allocator_(host, false)
        {
        }

        ~Impl()
        {
            for (uint32_t i = 0; i < static_cast<uint32_t>(CmdQueueType::Num); ++i)
            {
                host_.CpuWait(static_cast<CmdQueueType>(i));
            }

            mipmapper_.reset();

            read_back_mem_allocator_.Clear();
            upload_mem_allocator_.Clear();

            system_internal_.reset();
        }

        static Api SelectApi(Api api)
        {
            std::vector<Api> available_apis;
#ifdef AIHI_ENABLE_D3D12
            available_apis.push_back(Api::D3D12);
#endif
#ifdef AIHI_ENABLE_VULKAN
            available_apis.push_back(Api::Vulkan);
#endif

            if (api == Api::Auto)
            {
                return available_apis[0];
            }
            else
            {
                if (std::find(available_apis.begin(), available_apis.end(), api) != available_apis.end())
                {
                    return api;
                }
                else
                {
                    return available_apis[0];
                }
            }
        }

        GpuSystemInternal& Internal() noexcept
        {
            return *system_internal_;
        }

        GpuSystem::Api NativeApi() const noexcept
        {
            return api_;
        }

        CmdQueueType OverrideCmdQueueType(CmdQueueType type) const noexcept
        {
            if (!enable_async_compute_ && (type == CmdQueueType::Compute))
            {
                return CmdQueueType::Render;
            }
            return type;
        }

        GpuMemoryBlock AllocUploadMemBlock(uint32_t size_in_bytes, uint32_t alignment)
        {
            return upload_mem_allocator_.Allocate(size_in_bytes, alignment);
        }
        void DeallocUploadMemBlock(GpuMemoryBlock&& mem_block)
        {
            return upload_mem_allocator_.Deallocate(std::move(mem_block));
        }
        void ReallocUploadMemBlock(GpuMemoryBlock& mem_block, uint32_t size_in_bytes, uint32_t alignment)
        {
            return upload_mem_allocator_.Reallocate(mem_block, size_in_bytes, alignment);
        }

        GpuMemoryBlock AllocReadBackMemBlock(uint32_t size_in_bytes, uint32_t alignment)
        {
            return read_back_mem_allocator_.Allocate(size_in_bytes, alignment);
        }
        void DeallocReadBackMemBlock(GpuMemoryBlock&& mem_block)
        {
            return read_back_mem_allocator_.Deallocate(std::move(mem_block));
        }
        void ReallocReadBackMemBlock(GpuMemoryBlock& mem_block, uint32_t size_in_bytes, uint32_t alignment)
        {
            return read_back_mem_allocator_.Reallocate(mem_block, size_in_bytes, alignment);
        }

        GpuMipmapper& Mipmapper() noexcept
        {
            std::lock_guard lock(mipmapper_mutex_);
            if (!mipmapper_)
            {
                mipmapper_ = std::make_unique<GpuMipmapper>(host_);
            }
            return *mipmapper_;
        }

        void HandleDeviceLost()
        {
            upload_mem_allocator_.Clear();
            read_back_mem_allocator_.Clear();

            system_internal_->HandleDeviceLost();
        }

        void ClearStallResources()
        {
            system_internal_->ClearStallResources();

            for (uint32_t i = 0; i < static_cast<uint32_t>(CmdQueueType::Num); ++i)
            {
                const auto queue_type = static_cast<CmdQueueType>(i);
                const uint64_t completed_fence = system_internal_->CompletedFenceValue(queue_type);
                if (completed_fence != 0)
                {
                    upload_mem_allocator_.ClearStallPages(queue_type, completed_fence);
                    read_back_mem_allocator_.ClearStallPages(queue_type, completed_fence);
                }
            }
        }

    private:
        GpuSystem& host_;

        Api api_;
        bool enable_async_compute_;

        std::unique_ptr<GpuSystemInternal> system_internal_;

        GpuMemoryAllocator upload_mem_allocator_;
        GpuMemoryAllocator read_back_mem_allocator_;

        std::mutex mipmapper_mutex_;
        std::unique_ptr<GpuMipmapper> mipmapper_;
    };

    GpuSystem::GpuSystem(Api api, std::function<bool(Api api, void* device)> confirm_device, bool enable_sharing, bool enable_debug,
        bool enable_async_compute)
        : impl_(std::make_unique<Impl>(api, *this, std::move(confirm_device), enable_sharing, enable_debug, enable_async_compute))
    {
    }

    GpuSystem::~GpuSystem() = default;

    GpuSystem::GpuSystem(GpuSystem&& other) noexcept = default;
    GpuSystem& GpuSystem::operator=(GpuSystem&& other) noexcept = default;

    GpuSystem::Api GpuSystem::NativeApi() const noexcept
    {
        return impl_->NativeApi();
    }

    void* GpuSystem::NativeDevice() const noexcept
    {
        return impl_ ? impl_->Internal().NativeDevice() : nullptr;
    }

    void* GpuSystem::NativeCommandQueue(CmdQueueType type) const noexcept
    {
        return impl_ ? impl_->Internal().NativeCommandQueue(impl_->OverrideCmdQueueType(type)) : nullptr;
    }

    LUID GpuSystem::DeviceLuid() const noexcept
    {
        assert(impl_);
        return impl_->Internal().DeviceLuid();
    }

    void* GpuSystem::SharedFenceHandle(CmdQueueType type) const noexcept
    {
        return impl_ ? impl_->Internal().SharedFenceHandle(impl_->OverrideCmdQueueType(type)) : nullptr;
    }

    GpuCommandList GpuSystem::CreateCommandList(CmdQueueType type)
    {
        assert(impl_);
        return impl_->Internal().CreateCommandList(impl_->OverrideCmdQueueType(type));
    }

    uint64_t GpuSystem::Execute(GpuCommandList&& cmd_list, CmdQueueType wait_queue_type, uint64_t wait_fence_value)
    {
        assert(impl_);
        return impl_->Internal().Execute(std::move(cmd_list), impl_->OverrideCmdQueueType(wait_queue_type), wait_fence_value);
    }

    uint64_t GpuSystem::ExecuteAndReset(GpuCommandList& cmd_list, CmdQueueType wait_queue_type, uint64_t wait_fence_value)
    {
        assert(impl_);
        return impl_->Internal().ExecuteAndReset(cmd_list, impl_->OverrideCmdQueueType(wait_queue_type), wait_fence_value);
    }

    uint32_t GpuSystem::ConstantDataAlignment() const noexcept
    {
        assert(impl_);
        return impl_->Internal().ConstantDataAlignment();
    }
    uint32_t GpuSystem::StructuredDataAlignment() const noexcept
    {
        assert(impl_);
        return impl_->Internal().StructuredDataAlignment();
    }
    uint32_t GpuSystem::TextureDataAlignment() const noexcept
    {
        assert(impl_);
        return impl_->Internal().TextureDataAlignment();
    }

    GpuMemoryBlock GpuSystem::AllocUploadMemBlock(uint32_t size_in_bytes, uint32_t alignment)
    {
        assert(impl_);
        return impl_->AllocUploadMemBlock(size_in_bytes, alignment);
    }
    void GpuSystem::DeallocUploadMemBlock(GpuMemoryBlock&& mem_block)
    {
        assert(impl_);
        impl_->DeallocUploadMemBlock(std::move(mem_block));
    }
    void GpuSystem::ReallocUploadMemBlock(GpuMemoryBlock& mem_block, uint32_t size_in_bytes, uint32_t alignment)
    {
        assert(impl_);
        impl_->ReallocUploadMemBlock(mem_block, size_in_bytes, alignment);
    }

    GpuMemoryBlock GpuSystem::AllocReadBackMemBlock(uint32_t size_in_bytes, uint32_t alignment)
    {
        assert(impl_);
        return impl_->AllocReadBackMemBlock(size_in_bytes, alignment);
    }
    void GpuSystem::DeallocReadBackMemBlock(GpuMemoryBlock&& mem_block)
    {
        assert(impl_);
        impl_->DeallocReadBackMemBlock(std::move(mem_block));
    }
    void GpuSystem::ReallocReadBackMemBlock(GpuMemoryBlock& mem_block, uint32_t size_in_bytes, uint32_t alignment)
    {
        assert(impl_);
        impl_->ReallocReadBackMemBlock(mem_block, size_in_bytes, alignment);
    }

    void GpuSystem::CpuWait(CmdQueueType wait_queue_type, uint64_t wait_fence_value)
    {
        assert(impl_);
        impl_->Internal().CpuWait(impl_->OverrideCmdQueueType(wait_queue_type), wait_fence_value);
    }

    void GpuSystem::GpuWait(CmdQueueType target_queue_type, CmdQueueType wait_queue_type, uint64_t wait_fence_value)
    {
        assert(impl_);
        impl_->Internal().GpuWait(
            impl_->OverrideCmdQueueType(target_queue_type), impl_->OverrideCmdQueueType(wait_queue_type), wait_fence_value);
    }

    uint64_t GpuSystem::FenceValue(CmdQueueType type) const noexcept
    {
        assert(impl_);
        return impl_->Internal().FenceValue(impl_->OverrideCmdQueueType(type));
    }

    void GpuSystem::HandleDeviceLost()
    {
        assert(impl_);
        impl_->Internal().HandleDeviceLost();
    }

    GpuMipmapper& GpuSystem::Mipmapper() noexcept
    {
        assert(impl_);
        return impl_->Mipmapper();
    }

    GpuSystemInternal& GpuSystem::Internal() noexcept
    {
        assert(impl_);
        return impl_->Internal();
    }

    const GpuSystemInternal& GpuSystem::Internal() const noexcept
    {
        return const_cast<GpuSystem&>(*this).Internal();
    }
} // namespace AIHoloImager
