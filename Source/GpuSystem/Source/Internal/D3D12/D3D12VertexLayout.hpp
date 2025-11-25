// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <span>
#include <vector>

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Gpu/GpuVertexLayout.hpp"

#include "../GpuVertexLayoutInternal.hpp"
#include "D3D12ImpDefine.hpp"

namespace AIHoloImager
{
    class D3D12VertexLayout : public GpuVertexLayoutInternal
    {
    public:
        explicit D3D12VertexLayout(std::span<const GpuVertexAttrib> attribs, std::span<const uint32_t> slot_strides = {});
        ~D3D12VertexLayout() override;

        D3D12VertexLayout(const D3D12VertexLayout& other);
        explicit D3D12VertexLayout(const GpuVertexLayoutInternal& other);

        D3D12VertexLayout& operator=(const D3D12VertexLayout& other);
        GpuVertexLayoutInternal& operator=(const GpuVertexLayoutInternal& other) override;

        D3D12VertexLayout(D3D12VertexLayout&& other) noexcept;
        explicit D3D12VertexLayout(GpuVertexLayoutInternal&& other) noexcept;

        D3D12VertexLayout& operator=(D3D12VertexLayout&& other) noexcept;
        GpuVertexLayoutInternal& operator=(GpuVertexLayoutInternal&& other) noexcept override;

        std::unique_ptr<GpuVertexLayoutInternal> Clone() const override;

        std::span<const D3D12_INPUT_ELEMENT_DESC> InputElementDescs() const noexcept;
        std::span<const uint32_t> SlotStrides() const noexcept;

    private:
        void UpdateSemantics();

    private:
        std::vector<D3D12_INPUT_ELEMENT_DESC> input_elems_;
        std::vector<std::string> semantics_;
        std::vector<uint32_t> slot_strides_;
    };

    D3D12_DEFINE_IMP(VertexLayout)
} // namespace AIHoloImager
