// Copyright (c) 2025 Minmin Gong
//

#include "D3D12VertexAttrib.hpp"

#include <map>

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Gpu/GpuFormat.hpp"

#include "D3D12Conversion.hpp"

namespace AIHoloImager
{
    D3D12_IMP_IMP(VertexAttribs)

    D3D12VertexAttribs::D3D12VertexAttribs(std::span<const GpuVertexAttrib> attribs)
        : input_elems_(attribs.size()), semantics_(attribs.size())
    {
        std::map<uint32_t, uint32_t> slot_size;
        for (size_t i = 0; i < attribs.size(); ++i)
        {
            semantics_[i] = attribs[i].semantic;

            input_elems_[i].SemanticIndex = attribs[i].semantic_index;
            input_elems_[i].Format = ToDxgiFormat(attribs[i].format);
            input_elems_[i].InputSlot = attribs[i].slot;
            input_elems_[i].InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
            input_elems_[i].InstanceDataStepRate = 0;

            if (attribs[i].offset == GpuVertexAttrib::AppendOffset)
            {
                auto iter = slot_size.find(attribs[i].slot);
                if (iter == slot_size.end())
                {
                    iter = slot_size.emplace(attribs[i].slot, 0).first;
                }

                input_elems_[i].AlignedByteOffset = iter->second;
                iter->second += FormatSize(attribs[i].format);
            }
            else
            {
                input_elems_[i].AlignedByteOffset = attribs[i].offset;
            }
        }

        this->UpdateSemantics();
    }

    D3D12VertexAttribs::~D3D12VertexAttribs() = default;

    D3D12VertexAttribs::D3D12VertexAttribs(const D3D12VertexAttribs& other) : input_elems_(other.input_elems_), semantics_(other.semantics_)
    {
        this->UpdateSemantics();
    }
    D3D12VertexAttribs::D3D12VertexAttribs(const GpuVertexAttribsInternal& other)
        : D3D12VertexAttribs(static_cast<const D3D12VertexAttribs&>(other))
    {
    }

    D3D12VertexAttribs& D3D12VertexAttribs::operator=(const D3D12VertexAttribs& other)
    {
        if (this != &other)
        {
            input_elems_ = other.input_elems_;
            semantics_ = other.semantics_;

            this->UpdateSemantics();
        }
        return *this;
    }
    GpuVertexAttribsInternal& D3D12VertexAttribs::operator=(const GpuVertexAttribsInternal& other)
    {
        return this->operator=(static_cast<const D3D12VertexAttribs&>(other));
    }

    D3D12VertexAttribs::D3D12VertexAttribs(D3D12VertexAttribs&& other) noexcept
        : input_elems_(std::move(other.input_elems_)), semantics_(std::move(other.semantics_))
    {
        this->UpdateSemantics();
    }
    D3D12VertexAttribs::D3D12VertexAttribs(GpuVertexAttribsInternal&& other) noexcept
        : D3D12VertexAttribs(static_cast<D3D12VertexAttribs&&>(other))
    {
    }

    D3D12VertexAttribs& D3D12VertexAttribs::operator=(D3D12VertexAttribs&& other) noexcept
    {
        if (this != &other)
        {
            input_elems_ = std::move(other.input_elems_);
            semantics_ = std::move(other.semantics_);

            this->UpdateSemantics();
        }
        return *this;
    }
    GpuVertexAttribsInternal& D3D12VertexAttribs::operator=(GpuVertexAttribsInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12VertexAttribs&&>(other));
    }

    std::unique_ptr<GpuVertexAttribsInternal> D3D12VertexAttribs::Clone() const
    {
        return std::make_unique<D3D12VertexAttribs>(*this);
    }

    std::span<const D3D12_INPUT_ELEMENT_DESC> D3D12VertexAttribs::InputElementDescs() const
    {
        return std::span<const D3D12_INPUT_ELEMENT_DESC>(input_elems_);
    }

    void D3D12VertexAttribs::UpdateSemantics()
    {
        for (size_t i = 0; i < input_elems_.size(); ++i)
        {
            input_elems_[i].SemanticName = semantics_[i].c_str();
        }
    }
} // namespace AIHoloImager
