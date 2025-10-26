// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Gpu/GpuBuffer.hpp"
#include "Gpu/GpuCommandList.hpp"

#include "../GpuBufferInternal.hpp"
#include "D3D12CommandList.hpp"
#include "D3D12ImpDefine.hpp"
#include "D3D12Resource.hpp"

namespace AIHoloImager
{
    class D3D12Buffer : public GpuBufferInternal, public D3D12Resource
    {
    public:
        D3D12Buffer(GpuSystem& gpu_system, uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::wstring_view name);
        D3D12Buffer(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::wstring_view name);
        ~D3D12Buffer() override;

        D3D12Buffer(D3D12Buffer&& other) noexcept;
        explicit D3D12Buffer(GpuResourceInternal&& other) noexcept;
        explicit D3D12Buffer(GpuBufferInternal&& other) noexcept;
        D3D12Buffer& operator=(D3D12Buffer&& other) noexcept;
        GpuResourceInternal& operator=(GpuResourceInternal&& other) noexcept override;
        GpuBufferInternal& operator=(GpuBufferInternal&& other) noexcept override;

        void Name(std::wstring_view name) override;

        ID3D12Resource* Resource() const noexcept;
        void* NativeResource() const noexcept override;
        void* NativeBuffer() const noexcept override;

        void* SharedHandle() const noexcept override;

        GpuResourceType Type() const noexcept override;

        GpuVirtualAddressType GpuVirtualAddress() const noexcept override;
        uint32_t Size() const noexcept override;

        void* Map(const GpuRange& read_range) override;
        void* Map() override;
        void Unmap(const GpuRange& write_range) override;
        void Unmap() override;

        void Reset() override;

        void Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const override;
        void Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const override;
        void Transition(D3D12CommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const override;
        void Transition(D3D12CommandList& cmd_list, GpuResourceState target_state) const override;

    private:
        GpuHeap heap_{};
        mutable GpuResourceState curr_state_{};
    };

    D3D12_DEFINE_IMP(Buffer)
} // namespace AIHoloImager
