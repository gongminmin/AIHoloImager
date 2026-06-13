// Copyright (c) 2025-2026 Minmin Gong
//

#include "D3D12VertexLayout.hpp"

#include <map>

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Gpu/GpuFormat.hpp"

#include "D3D12Conversion.hpp"

namespace AIHoloImager
{
    D3D12_IMP_IMP(VertexLayout)

    D3D12VertexLayout::D3D12VertexLayout(std::span<const GpuVertexAttrib> attribs, std::span<const uint32_t> slot_strides)
        : attribs_(attribs.begin(), attribs.end()), input_elems_(attribs.size()), semantics_(attribs.size())
    {
        uint32_t max_slot = 0;
        std::map<uint32_t, uint32_t> slot_size;
        for (size_t i = 0; i < attribs.size(); ++i)
        {
            semantics_[i] = attribs[i].semantic;

            input_elems_[i].SemanticIndex = attribs[i].semantic_index;
            input_elems_[i].Format = ToDxgiFormat(attribs[i].format);
            input_elems_[i].InputSlot = attribs[i].slot;
            input_elems_[i].InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
            input_elems_[i].InstanceDataStepRate = 0;

            max_slot = std::max(max_slot, attribs[i].slot);

            auto iter = slot_size.find(attribs[i].slot);
            if (iter == slot_size.end())
            {
                iter = slot_size.emplace(attribs[i].slot, 0).first;
            }
            if (attribs[i].offset == GpuVertexAttrib::AppendOffset)
            {
                input_elems_[i].AlignedByteOffset = iter->second;
            }
            else
            {
                input_elems_[i].AlignedByteOffset = attribs[i].offset;
            }
            iter->second = std::max(iter->second, input_elems_[i].AlignedByteOffset + FormatSize(attribs[i].format));

            attribs_[i].offset = input_elems_[i].AlignedByteOffset;
        }

        if (slot_strides.empty())
        {
            slot_strides_.resize(max_slot + 1, 0);
            for (const auto& [slot, size] : slot_size)
            {
                slot_strides_[slot] = size;
            }
        }
        else
        {
            slot_strides_ = std::vector(slot_strides.begin(), slot_strides.end());
        }

        this->UpdateSemantics();
    }

    D3D12VertexLayout::~D3D12VertexLayout() = default;

    D3D12VertexLayout::D3D12VertexLayout(const D3D12VertexLayout& other)
        : attribs_(other.attribs_), input_elems_(other.input_elems_), semantics_(other.semantics_), slot_strides_(other.slot_strides_)
    {
        this->UpdateSemantics();
    }
    D3D12VertexLayout::D3D12VertexLayout(const GpuVertexLayoutInternal& other)
        : D3D12VertexLayout(static_cast<const D3D12VertexLayout&>(other))
    {
    }

    D3D12VertexLayout& D3D12VertexLayout::operator=(const D3D12VertexLayout& other)
    {
        if (this != &other)
        {
            attribs_ = other.attribs_;
            input_elems_ = other.input_elems_;
            semantics_ = other.semantics_;
            slot_strides_ = other.slot_strides_;

            this->UpdateSemantics();
        }
        return *this;
    }
    GpuVertexLayoutInternal& D3D12VertexLayout::operator=(const GpuVertexLayoutInternal& other)
    {
        return this->operator=(static_cast<const D3D12VertexLayout&>(other));
    }

    D3D12VertexLayout::D3D12VertexLayout(D3D12VertexLayout&& other) noexcept
        : attribs_(std::move(other.attribs_)), input_elems_(std::move(other.input_elems_)), semantics_(std::move(other.semantics_)),
          slot_strides_(std::move(other.slot_strides_))
    {
        this->UpdateSemantics();
    }
    D3D12VertexLayout::D3D12VertexLayout(GpuVertexLayoutInternal&& other) noexcept
        : D3D12VertexLayout(static_cast<D3D12VertexLayout&&>(other))
    {
    }

    D3D12VertexLayout& D3D12VertexLayout::operator=(D3D12VertexLayout&& other) noexcept
    {
        if (this != &other)
        {
            attribs_ = std::move(other.attribs_);
            input_elems_ = std::move(other.input_elems_);
            semantics_ = std::move(other.semantics_);
            slot_strides_ = std::move(other.slot_strides_);

            this->UpdateSemantics();
        }
        return *this;
    }
    GpuVertexLayoutInternal& D3D12VertexLayout::operator=(GpuVertexLayoutInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12VertexLayout&&>(other));
    }

    std::unique_ptr<GpuVertexLayoutInternal> D3D12VertexLayout::Clone() const
    {
        return std::make_unique<D3D12VertexLayout>(*this);
    }

    std::span<const GpuVertexAttrib> D3D12VertexLayout::Attribs() const noexcept
    {
        return attribs_;
    }

    std::span<const uint32_t> D3D12VertexLayout::SlotStrides() const noexcept
    {
        return slot_strides_;
    }

    std::span<const D3D12_INPUT_ELEMENT_DESC> D3D12VertexLayout::InputElementDescs() const noexcept
    {
        return input_elems_;
    }

    void D3D12VertexLayout::UpdateSemantics()
    {
        for (size_t i = 0; i < input_elems_.size(); ++i)
        {
            input_elems_[i].SemanticName = semantics_[i].c_str();
        }
    }
} // namespace AIHoloImager
