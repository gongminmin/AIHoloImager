// Copyright (c) 2024 Minmin Gong
//

#include "GpuVertexAttrib.hpp"

#include <map>

#include "GpuFormat.hpp"

namespace AIHoloImager
{
    GpuVertexAttribs::GpuVertexAttribs(std::span<const GpuVertexAttrib> attribs) : input_elems_(attribs.size()), semantics_(attribs.size())
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

    GpuVertexAttribs::GpuVertexAttribs(const GpuVertexAttribs& other) noexcept
        : input_elems_(other.input_elems_), semantics_(other.semantics_)
    {
        this->UpdateSemantics();
    }

    GpuVertexAttribs& GpuVertexAttribs::operator=(const GpuVertexAttribs& other) noexcept
    {
        if (this != &other)
        {
            input_elems_ = other.input_elems_;
            semantics_ = other.semantics_;

            this->UpdateSemantics();
        }
        return *this;
    }

    GpuVertexAttribs::GpuVertexAttribs(GpuVertexAttribs&& other) noexcept
        : input_elems_(std::move(other.input_elems_)), semantics_(std::move(other.semantics_))
    {
        this->UpdateSemantics();
    }

    GpuVertexAttribs& GpuVertexAttribs::operator=(GpuVertexAttribs&& other) noexcept
    {
        if (this != &other)
        {
            input_elems_ = std::move(other.input_elems_);
            semantics_ = std::move(other.semantics_);

            this->UpdateSemantics();
        }
        return *this;
    }

    void GpuVertexAttribs::UpdateSemantics()
    {
        for (size_t i = 0; i < input_elems_.size(); ++i)
        {
            input_elems_[i].SemanticName = semantics_[i].c_str();
        }
    }

    std::span<const D3D12_INPUT_ELEMENT_DESC> GpuVertexAttribs::InputElementDescs() const
    {
        return std::span<const D3D12_INPUT_ELEMENT_DESC>(input_elems_);
    }
} // namespace AIHoloImager
