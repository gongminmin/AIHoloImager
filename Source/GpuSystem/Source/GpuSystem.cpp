// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuSystem.hpp"

#include <cassert>
#include <format>
#include <iterator>
#include <mutex>

#include "Gpu/GpuCommandList.hpp"

#include "Internal/GpuSystemInternal.hpp"
#include "Internal/GpuSystemInternalFactory.hpp"

namespace AIHoloImager
{
    class GpuSystem::Impl
    {
    public:
        Impl(Api api, GpuSystem& host, std::function<bool(Api api, void* device)> confirm_device, bool enable_sharing, bool enable_debug)
            : host_(host), api_(SelectApi(api)),
              system_internal_(CreateGpuSystemInternal(api_, host, std::move(confirm_device), enable_sharing, enable_debug)),
              upload_mem_allocator_(host, true), read_back_mem_allocator_(host, false)
        {
        }

        ~Impl()
        {
            host_.CpuWait();

            mipmapper_.reset();

            read_back_mem_allocator_.Clear();
            upload_mem_allocator_.Clear();

            system_internal_.reset();
        }

        static Api SelectApi(Api api)
        {
            const Api available_apis[] = {Api::D3D12};

            if (api == Api::Auto)
            {
                return available_apis[0];
            }
            else
            {
                if (std::find(std::begin(available_apis), std::end(available_apis), api) != std::end(available_apis))
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

        GpuMemoryBlock AllocUploadMemBlock(uint32_t size_in_bytes, uint32_t alignment)
        {
            return upload_mem_allocator_.Allocate(size_in_bytes, alignment);
        }
        void DeallocUploadMemBlock(GpuMemoryBlock&& mem_block)
        {
            return upload_mem_allocator_.Deallocate(std::move(mem_block), system_internal_->FenceValue());
        }
        void ReallocUploadMemBlock(GpuMemoryBlock& mem_block, uint32_t size_in_bytes, uint32_t alignment)
        {
            return upload_mem_allocator_.Reallocate(mem_block, system_internal_->FenceValue(), size_in_bytes, alignment);
        }

        GpuMemoryBlock AllocReadBackMemBlock(uint32_t size_in_bytes, uint32_t alignment)
        {
            return read_back_mem_allocator_.Allocate(size_in_bytes, alignment);
        }
        void DeallocReadBackMemBlock(GpuMemoryBlock&& mem_block)
        {
            return read_back_mem_allocator_.Deallocate(std::move(mem_block), system_internal_->FenceValue());
        }
        void ReallocReadBackMemBlock(GpuMemoryBlock& mem_block, uint32_t size_in_bytes, uint32_t alignment)
        {
            return read_back_mem_allocator_.Reallocate(mem_block, system_internal_->FenceValue(), size_in_bytes, alignment);
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

            const uint64_t completed_fence = system_internal_->CompletedFenceValue();

            upload_mem_allocator_.ClearStallPages(completed_fence);
            read_back_mem_allocator_.ClearStallPages(completed_fence);
        }

    private:
        GpuSystem& host_;

        Api api_;

        std::unique_ptr<GpuSystemInternal> system_internal_;

        GpuMemoryAllocator upload_mem_allocator_;
        GpuMemoryAllocator read_back_mem_allocator_;

        std::mutex mipmapper_mutex_;
        std::unique_ptr<GpuMipmapper> mipmapper_;
    };

    GpuSystem::GpuSystem(Api api, std::function<bool(Api api, void* device)> confirm_device, bool enable_sharing, bool enable_debug)
        : impl_(std::make_unique<Impl>(api, *this, std::move(confirm_device), enable_sharing, enable_debug))
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
        return impl_ ? impl_->Internal().NativeCommandQueue(type) : nullptr;
    }

    LUID GpuSystem::DeviceLuid() const noexcept
    {
        assert(impl_);
        return impl_->Internal().DeviceLuid();
    }

    void* GpuSystem::SharedFenceHandle() const noexcept
    {
        return impl_ ? impl_->Internal().SharedFenceHandle() : nullptr;
    }

    GpuCommandList GpuSystem::CreateCommandList(GpuSystem::CmdQueueType type)
    {
        assert(impl_);
        return impl_->Internal().CreateCommandList(type);
    }

    uint64_t GpuSystem::Execute(GpuCommandList&& cmd_list, uint64_t wait_fence_value)
    {
        assert(impl_);
        return impl_->Internal().Execute(std::move(cmd_list), wait_fence_value);
    }

    uint64_t GpuSystem::ExecuteAndReset(GpuCommandList& cmd_list, uint64_t wait_fence_value)
    {
        assert(impl_);
        return impl_->Internal().ExecuteAndReset(cmd_list, wait_fence_value);
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

    void GpuSystem::CpuWait(uint64_t fence_value)
    {
        assert(impl_);
        impl_->Internal().CpuWait(fence_value);
    }

    void GpuSystem::GpuWait(CmdQueueType type, uint64_t fence_value)
    {
        assert(impl_);
        impl_->Internal().GpuWait(type, fence_value);
    }

    uint64_t GpuSystem::FenceValue() const noexcept
    {
        assert(impl_);
        return impl_->Internal().FenceValue();
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
