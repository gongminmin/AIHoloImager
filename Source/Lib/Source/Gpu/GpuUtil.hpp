// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class GpuSystem;

    template <typename T>
    class GpuRecyclableObject
    {
        DISALLOW_COPY_AND_ASSIGN(GpuRecyclableObject)

    public:
        GpuRecyclableObject() = default;
        GpuRecyclableObject(GpuSystem& gpu_system, T object) : gpu_system_(&gpu_system), object_(std::move(object))
        {
        }

        ~GpuRecyclableObject()
        {
            this->Recycle();
        }

        GpuRecyclableObject(GpuRecyclableObject&& other) noexcept = default;
        GpuRecyclableObject& operator=(GpuRecyclableObject&& other) noexcept
        {
            if (this != &other)
            {
                this->Recycle();

                gpu_system_ = std::exchange(other.gpu_system_, nullptr);
                object_ = std::move(other.object_);
            }

            return *this;
        }

        GpuRecyclableObject Share() const
        {
            return GpuRecyclableObject(const_cast<GpuSystem&>(*gpu_system_), object_);
        }

        GpuSystem* GpuSys() noexcept
        {
            return gpu_system_;
        }
        const GpuSystem* GpuSys() const noexcept
        {
            return gpu_system_;
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
            gpu_system_ = nullptr;
            object_ = nullptr;
        }

    private:
        void Recycle()
        {
            if (object_ && object_.Unique())
            {
                gpu_system_->Recycle(std::move(object_));
            }
        }

    private:
        GpuSystem* gpu_system_ = nullptr;
        T object_;
    };
} // namespace AIHoloImager
