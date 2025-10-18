// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include "Base/Noncopyable.hpp"

namespace AIHoloImager
{
    class D3D12System;

    template <typename T>
    class D3D12RecyclableObject
    {
        DISALLOW_COPY_AND_ASSIGN(D3D12RecyclableObject)

    public:
        D3D12RecyclableObject() = default;
        D3D12RecyclableObject(D3D12System& d3d12_system, T object) : d3d12_system_(&d3d12_system), object_(std::move(object))
        {
        }

        ~D3D12RecyclableObject()
        {
            this->Recycle();
        }

        D3D12RecyclableObject(D3D12RecyclableObject&& other) noexcept = default;
        D3D12RecyclableObject& operator=(D3D12RecyclableObject&& other) noexcept
        {
            if (this != &other)
            {
                this->Recycle();

                d3d12_system_ = std::exchange(other.d3d12_system_, {});
                object_ = std::move(other.object_);
            }

            return *this;
        }

        D3D12System* D3D12Sys() noexcept
        {
            return d3d12_system_;
        }
        const D3D12System* D3D12Sys() const noexcept
        {
            return d3d12_system_;
        }

        T& Object() noexcept
        {
            return object_;
        }
        const T& Object() const noexcept
        {
            return object_;
        }

        explicit operator bool() const noexcept
        {
            return static_cast<bool>(object_);
        }

        auto operator->() noexcept
        {
            return object_.Get();
        }
        auto operator->() const noexcept
        {
            return object_.Get();
        }

        void Reset()
        {
            this->Recycle();
            d3d12_system_ = nullptr;
            object_ = nullptr;
        }

    private:
        void Recycle()
        {
            if (object_ && object_.Unique())
            {
                d3d12_system_->Recycle(std::move(object_));
            }
        }

    private:
        D3D12System* d3d12_system_ = nullptr;
        T object_;
    };
} // namespace AIHoloImager
