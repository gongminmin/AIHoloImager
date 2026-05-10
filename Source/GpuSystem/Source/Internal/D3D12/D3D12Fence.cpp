// Copyright (c) 2026 Minmin Gong
//

#include "D3D12Fence.hpp"

#include "D3D12System.hpp"

DEFINE_UUID_OF(ID3D12Fence);

namespace AIHoloImager
{
    D3D12_IMP_IMP(Fence)

    D3D12Fence::D3D12Fence() noexcept = default;
    D3D12Fence::D3D12Fence(GpuSystem& gpu_system, uint64_t init_val, bool enable_sharing)
    {
        ID3D12Device* d3d12_device = D3D12Imp(gpu_system).Device();

        TIFHR(d3d12_device->CreateFence(
            init_val, enable_sharing ? D3D12_FENCE_FLAG_SHARED : D3D12_FENCE_FLAG_NONE, UuidOf<ID3D12Fence>(), fence_.PutVoid()));

        fence_event_ = MakeWin32UniqueHandle(::CreateEvent(nullptr, FALSE, FALSE, nullptr));
        Verify(fence_event_.get() != INVALID_HANDLE_VALUE);

        if (enable_sharing)
        {
            HANDLE shared_handle;
            TIFHR(d3d12_device->CreateSharedHandle(fence_.Get(), nullptr, GENERIC_ALL, nullptr, &shared_handle));
            shared_fence_handle_.reset(shared_handle);
        }
    }

    D3D12Fence::D3D12Fence(D3D12Fence&& other) noexcept = default;
    D3D12Fence::D3D12Fence(GpuFenceInternal&& other) noexcept : D3D12Fence(static_cast<D3D12Fence&&>(other))
    {
    }

    D3D12Fence::~D3D12Fence() noexcept = default;

    D3D12Fence& D3D12Fence::operator=(D3D12Fence&& other) noexcept = default;
    GpuFenceInternal& D3D12Fence::operator=(GpuFenceInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12Fence&&>(other));
    }

    void* D3D12Fence::NativeFence() const noexcept
    {
        return this->Fence();
    }

    void* D3D12Fence::SharedFenceHandle() const noexcept
    {
        return shared_fence_handle_.get();
    }

    uint64_t D3D12Fence::CompletedValue() const
    {
        return fence_ ? fence_->GetCompletedValue() : 0;
    }

    void D3D12Fence::CpuWait(uint64_t value) const
    {
        if (SUCCEEDED(fence_->SetEventOnCompletion(value, fence_event_.get())))
        {
            ::WaitForSingleObjectEx(fence_event_.get(), INFINITE, FALSE);
        }
    }

    ID3D12Fence* D3D12Fence::Fence() const noexcept
    {
        return fence_.Get();
    }
} // namespace AIHoloImager
