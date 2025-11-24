// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <string_view>

#include <volk.h>

#include "Base/Noncopyable.hpp"

namespace AIHoloImager
{
    class VulkanSystem;

    template <typename T>
    class VulkanRecyclableObject
    {
        DISALLOW_COPY_AND_ASSIGN(VulkanRecyclableObject)

    public:
        VulkanRecyclableObject() = default;
        VulkanRecyclableObject(VulkanSystem& vulkan_system, T object) : vulkan_system_(&vulkan_system), object_(object)
        {
        }

        ~VulkanRecyclableObject()
        {
            this->Recycle();
        }

        VulkanRecyclableObject(VulkanRecyclableObject&& other) noexcept
            : vulkan_system_(std::exchange(other.vulkan_system_, {})), object_(std::exchange(other.object_, VK_NULL_HANDLE)),
              extra_recycle_param_(std::exchange(other.extra_recycle_param_, nullptr))
        {
        }
        VulkanRecyclableObject& operator=(VulkanRecyclableObject&& other) noexcept
        {
            if (this != &other)
            {
                this->Recycle();

                vulkan_system_ = std::exchange(other.vulkan_system_, {});
                object_ = std::exchange(other.object_, VK_NULL_HANDLE);
                extra_recycle_param_ = std::exchange(other.extra_recycle_param_, nullptr);
            }

            return *this;
        }

        VulkanSystem* VulkanSys() noexcept
        {
            return vulkan_system_;
        }
        const VulkanSystem* VulkanSys() const noexcept
        {
            return vulkan_system_;
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
            vulkan_system_ = nullptr;
            object_ = nullptr;
        }

        void AddExtraRecycleParam(void* extra_recycle_param)
        {
            extra_recycle_param_ = extra_recycle_param;
        }

    private:
        void Recycle()
        {
            if (object_ != VK_NULL_HANDLE)
            {
                if constexpr (std::is_same_v<T, VkDescriptorSet>)
                {
                    vulkan_system_->Recycle(object_, reinterpret_cast<VkDescriptorPool>(extra_recycle_param_));
                }
                else
                {
                    vulkan_system_->Recycle(object_);
                }
            }
        }

    private:
        VulkanSystem* vulkan_system_ = nullptr;
        T object_ = VK_NULL_HANDLE;
        void* extra_recycle_param_ = nullptr;
    };

    void SetName(const VulkanSystem& vulkan_system, const void* vulkan_object, VkObjectType type, std::string_view name);
} // namespace AIHoloImager
