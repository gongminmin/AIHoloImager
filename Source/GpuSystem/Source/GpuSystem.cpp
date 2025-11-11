// Copyright (c) 2024-2025 Minmin Gong
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
        Impl(Api api, GpuSystem& host, std::function<bool(Api api, void* device)> confirm_device, bool enable_sharing, bool enable_debug)
            : host_(host), system_internal_(CreateGpuSystemInternal(api, host, std::move(confirm_device), enable_sharing, enable_debug)),
              upload_mem_allocator_(host, true), read_back_mem_allocator_(host, false),
              rtv_desc_allocator_(host, GpuDescriptorHeapType::Rtv, false), dsv_desc_allocator_(host, GpuDescriptorHeapType::Dsv, false),
              cbv_srv_uav_desc_allocator_(host, GpuDescriptorHeapType::CbvSrvUav, false),
              shader_visible_cbv_srv_uav_desc_allocator_(host, GpuDescriptorHeapType::CbvSrvUav, true),
              sampler_desc_allocator_(host, GpuDescriptorHeapType::Sampler, false),
              shader_visible_sampler_desc_allocator_(host, GpuDescriptorHeapType::Sampler, true)
        {
        }

        ~Impl()
        {
            host_.CpuWait();

            mipmapper_.reset();

            rtv_desc_allocator_.Clear();
            shader_visible_sampler_desc_allocator_.Clear();
            sampler_desc_allocator_.Clear();
            shader_visible_cbv_srv_uav_desc_allocator_.Clear();
            cbv_srv_uav_desc_allocator_.Clear();
            dsv_desc_allocator_.Clear();

            read_back_mem_allocator_.Clear();
            upload_mem_allocator_.Clear();

            system_internal_.reset();
        }

        GpuSystemInternal& Internal() noexcept
        {
            return *system_internal_;
        }

        uint32_t RtvDescSize() const noexcept
        {
            return rtv_desc_allocator_.DescriptorSize();
        }
        uint32_t DsvDescSize() const noexcept
        {
            return dsv_desc_allocator_.DescriptorSize();
        }
        uint32_t CbvSrvUavDescSize() const noexcept
        {
            return cbv_srv_uav_desc_allocator_.DescriptorSize();
        }
        uint32_t SamplerDescSize() const noexcept
        {
            return sampler_desc_allocator_.DescriptorSize();
        }

        GpuDescriptorBlock AllocRtvDescBlock(uint32_t size)
        {
            return rtv_desc_allocator_.Allocate(size);
        }
        void DeallocRtvDescBlock(GpuDescriptorBlock&& desc_block)
        {
            return rtv_desc_allocator_.Deallocate(std::move(desc_block), system_internal_->FenceValue());
        }
        void ReallocRtvDescBlock(GpuDescriptorBlock& desc_block, uint32_t size)
        {
            return rtv_desc_allocator_.Reallocate(desc_block, system_internal_->FenceValue(), size);
        }

        GpuDescriptorBlock AllocDsvDescBlock(uint32_t size)
        {
            return dsv_desc_allocator_.Allocate(size);
        }
        void DeallocDsvDescBlock(GpuDescriptorBlock&& desc_block)
        {
            return dsv_desc_allocator_.Deallocate(std::move(desc_block), system_internal_->FenceValue());
        }
        void ReallocDsvDescBlock(GpuDescriptorBlock& desc_block, uint32_t size)
        {
            return dsv_desc_allocator_.Reallocate(desc_block, system_internal_->FenceValue(), size);
        }

        GpuDescriptorBlock AllocCbvSrvUavDescBlock(uint32_t size)
        {
            return cbv_srv_uav_desc_allocator_.Allocate(size);
        }
        void DeallocCbvSrvUavDescBlock(GpuDescriptorBlock&& desc_block)
        {
            return cbv_srv_uav_desc_allocator_.Deallocate(std::move(desc_block), system_internal_->FenceValue());
        }
        void ReallocCbvSrvUavDescBlock(GpuDescriptorBlock& desc_block, uint32_t size)
        {
            return cbv_srv_uav_desc_allocator_.Reallocate(desc_block, system_internal_->FenceValue(), size);
        }

        GpuDescriptorBlock AllocShaderVisibleCbvSrvUavDescBlock(uint32_t size)
        {
            return shader_visible_cbv_srv_uav_desc_allocator_.Allocate(size);
        }
        void DeallocShaderVisibleCbvSrvUavDescBlock(GpuDescriptorBlock&& desc_block)
        {
            return shader_visible_cbv_srv_uav_desc_allocator_.Deallocate(std::move(desc_block), system_internal_->FenceValue());
        }
        void ReallocShaderVisibleCbvSrvUavDescBlock(GpuDescriptorBlock& desc_block, uint32_t size)
        {
            return shader_visible_cbv_srv_uav_desc_allocator_.Reallocate(desc_block, system_internal_->FenceValue(), size);
        }

        GpuDescriptorBlock AllocSamplerDescBlock(uint32_t size)
        {
            return sampler_desc_allocator_.Allocate(size);
        }
        void DeallocSamplerDescBlock(GpuDescriptorBlock&& desc_block)
        {
            return sampler_desc_allocator_.Deallocate(std::move(desc_block), system_internal_->FenceValue());
        }
        void ReallocSamplerDescBlock(GpuDescriptorBlock& desc_block, uint32_t size)
        {
            return sampler_desc_allocator_.Reallocate(desc_block, system_internal_->FenceValue(), size);
        }

        GpuDescriptorBlock AllocShaderVisibleSamplerDescBlock(uint32_t size)
        {
            return shader_visible_sampler_desc_allocator_.Allocate(size);
        }
        void DeallocShaderVisibleSamplerDescBlock(GpuDescriptorBlock&& desc_block)
        {
            return shader_visible_sampler_desc_allocator_.Deallocate(std::move(desc_block), system_internal_->FenceValue());
        }
        void ReallocShaderVisibleSamplerDescBlock(GpuDescriptorBlock& desc_block, uint32_t size)
        {
            return shader_visible_sampler_desc_allocator_.Reallocate(desc_block, system_internal_->FenceValue(), size);
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

            rtv_desc_allocator_.Clear();
            dsv_desc_allocator_.Clear();
            cbv_srv_uav_desc_allocator_.Clear();
            shader_visible_cbv_srv_uav_desc_allocator_.Clear();
            sampler_desc_allocator_.Clear();
            shader_visible_sampler_desc_allocator_.Clear();

            system_internal_->HandleDeviceLost();
        }

        void ClearStallResources()
        {
            system_internal_->ClearStallResources();

            const uint64_t completed_fence = system_internal_->CompletedFenceValue();

            upload_mem_allocator_.ClearStallPages(completed_fence);
            read_back_mem_allocator_.ClearStallPages(completed_fence);

            rtv_desc_allocator_.ClearStallPages(completed_fence);
            dsv_desc_allocator_.ClearStallPages(completed_fence);
            cbv_srv_uav_desc_allocator_.ClearStallPages(completed_fence);
            shader_visible_cbv_srv_uav_desc_allocator_.ClearStallPages(completed_fence);
            sampler_desc_allocator_.ClearStallPages(completed_fence);
            shader_visible_sampler_desc_allocator_.ClearStallPages(completed_fence);
        }

    private:
        GpuSystem& host_;

        std::unique_ptr<GpuSystemInternal> system_internal_;

        GpuMemoryAllocator upload_mem_allocator_;
        GpuMemoryAllocator read_back_mem_allocator_;

        GpuDescriptorAllocator rtv_desc_allocator_;
        GpuDescriptorAllocator dsv_desc_allocator_;
        GpuDescriptorAllocator cbv_srv_uav_desc_allocator_;
        GpuDescriptorAllocator shader_visible_cbv_srv_uav_desc_allocator_;
        GpuDescriptorAllocator sampler_desc_allocator_;
        GpuDescriptorAllocator shader_visible_sampler_desc_allocator_;

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

    uint32_t GpuSystem::RtvDescSize() const noexcept
    {
        assert(impl_);
        return impl_->RtvDescSize();
    }
    uint32_t GpuSystem::DsvDescSize() const noexcept
    {
        assert(impl_);
        return impl_->DsvDescSize();
    }
    uint32_t GpuSystem::CbvSrvUavDescSize() const noexcept
    {
        assert(impl_);
        return impl_->CbvSrvUavDescSize();
    }
    uint32_t GpuSystem::SamplerDescSize() const noexcept
    {
        assert(impl_);
        return impl_->SamplerDescSize();
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

    GpuDescriptorBlock GpuSystem::AllocRtvDescBlock(uint32_t size)
    {
        assert(impl_);
        return impl_->AllocRtvDescBlock(size);
    }
    void GpuSystem::DeallocRtvDescBlock(GpuDescriptorBlock&& desc_block)
    {
        assert(impl_);
        impl_->DeallocRtvDescBlock(std::move(desc_block));
    }
    void GpuSystem::ReallocRtvDescBlock(GpuDescriptorBlock& desc_block, uint32_t size)
    {
        assert(impl_);
        impl_->ReallocRtvDescBlock(desc_block, size);
    }

    GpuDescriptorBlock GpuSystem::AllocDsvDescBlock(uint32_t size)
    {
        assert(impl_);
        return impl_->AllocDsvDescBlock(size);
    }
    void GpuSystem::DeallocDsvDescBlock(GpuDescriptorBlock&& desc_block)
    {
        assert(impl_);
        impl_->DeallocDsvDescBlock(std::move(desc_block));
    }
    void GpuSystem::ReallocDsvDescBlock(GpuDescriptorBlock& desc_block, uint32_t size)
    {
        assert(impl_);
        impl_->ReallocDsvDescBlock(desc_block, size);
    }

    GpuDescriptorBlock GpuSystem::AllocCbvSrvUavDescBlock(uint32_t size)
    {
        assert(impl_);
        return impl_->AllocCbvSrvUavDescBlock(size);
    }
    void GpuSystem::DeallocCbvSrvUavDescBlock(GpuDescriptorBlock&& desc_block)
    {
        assert(impl_);
        impl_->DeallocCbvSrvUavDescBlock(std::move(desc_block));
    }
    void GpuSystem::ReallocCbvSrvUavDescBlock(GpuDescriptorBlock& desc_block, uint32_t size)
    {
        assert(impl_);
        impl_->ReallocCbvSrvUavDescBlock(desc_block, size);
    }

    GpuDescriptorBlock GpuSystem::AllocShaderVisibleCbvSrvUavDescBlock(uint32_t size)
    {
        assert(impl_);
        return impl_->AllocShaderVisibleCbvSrvUavDescBlock(size);
    }
    void GpuSystem::DeallocShaderVisibleCbvSrvUavDescBlock(GpuDescriptorBlock&& desc_block)
    {
        assert(impl_);
        impl_->DeallocShaderVisibleCbvSrvUavDescBlock(std::move(desc_block));
    }
    void GpuSystem::ReallocShaderVisibleCbvSrvUavDescBlock(GpuDescriptorBlock& desc_block, uint32_t size)
    {
        assert(impl_);
        impl_->ReallocShaderVisibleCbvSrvUavDescBlock(desc_block, size);
    }

    GpuDescriptorBlock GpuSystem::AllocSamplerDescBlock(uint32_t size)
    {
        assert(impl_);
        return impl_->AllocSamplerDescBlock(size);
    }
    void GpuSystem::DeallocSamplerDescBlock(GpuDescriptorBlock&& desc_block)
    {
        assert(impl_);
        impl_->DeallocSamplerDescBlock(std::move(desc_block));
    }
    void GpuSystem::ReallocSamplerDescBlock(GpuDescriptorBlock& desc_block, uint32_t size)
    {
        assert(impl_);
        impl_->ReallocSamplerDescBlock(desc_block, size);
    }

    GpuDescriptorBlock GpuSystem::AllocShaderVisibleSamplerDescBlock(uint32_t size)
    {
        assert(impl_);
        return impl_->AllocShaderVisibleSamplerDescBlock(size);
    }
    void GpuSystem::DeallocShaderVisibleSamplerDescBlock(GpuDescriptorBlock&& desc_block)
    {
        assert(impl_);
        impl_->DeallocShaderVisibleSamplerDescBlock(std::move(desc_block));
    }
    void GpuSystem::ReallocShaderVisibleSamplerDescBlock(GpuDescriptorBlock& desc_block, uint32_t size)
    {
        assert(impl_);
        impl_->ReallocShaderVisibleSamplerDescBlock(desc_block, size);
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
