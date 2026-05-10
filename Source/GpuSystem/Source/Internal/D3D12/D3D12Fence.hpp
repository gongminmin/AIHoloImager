// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Base/ComPtr.hpp"
#include "Base/SmartPtrHelper.hpp"
#include "Gpu/GpuFence.hpp"
#include "Gpu/GpuSystem.hpp"

#include "../GpuFenceInternal.hpp"
#include "D3D12ImpDefine.hpp"

namespace AIHoloImager
{
    class D3D12Fence final : public GpuFenceInternal
    {
        DISALLOW_COPY_AND_ASSIGN(D3D12Fence)

    public:
        D3D12Fence() noexcept;
        D3D12Fence(GpuSystem& gpu_system, uint64_t init_val, bool enable_sharing);
        ~D3D12Fence() noexcept override;

        D3D12Fence(D3D12Fence&& other) noexcept;
        explicit D3D12Fence(GpuFenceInternal&& other) noexcept;
        D3D12Fence& operator=(D3D12Fence&& other) noexcept;
        GpuFenceInternal& operator=(GpuFenceInternal&& other) noexcept override;

        void* NativeFence() const noexcept override;
        void* SharedFenceHandle() const noexcept override;

        uint64_t CompletedValue() const override;

        void CpuWait(uint64_t value) const override;

        ID3D12Fence* Fence() const noexcept;

    private:
        ComPtr<ID3D12Fence> fence_;
        Win32UniqueHandle fence_event_;
        Win32UniqueHandle shared_fence_handle_;
    };

    D3D12_DEFINE_IMP(Fence)
} // namespace AIHoloImager
