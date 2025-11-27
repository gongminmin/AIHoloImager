// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <stdexcept>

#include <spirv_reflect.h>
#include <volk.h>

#include "Base/ErrorHandling.hpp"

namespace AIHoloImager
{
    template <typename ResultType>
    class VulkanErrorException : public std::runtime_error
    {
    public:
        VulkanErrorException(ResultType result, std::string_view file, uint32_t line)
            : std::runtime_error(CombineFileLine(result, std::move(file), line)), result_(result)
        {
        }

        ResultType Result() const noexcept
        {
            return result_;
        }

    private:
        const ResultType result_;
    };
} // namespace AIHoloImager

#define TIFVK(x)                                                                                  \
    {                                                                                             \
        const auto inner_result = (x);                                                            \
        if (inner_result != VK_SUCCESS)                                                           \
        {                                                                                         \
            throw AIHoloImager::VulkanErrorException<VkResult>(inner_result, __FILE__, __LINE__); \
        }                                                                                         \
    }

#define TIFSPVRFL(x)                                                                                      \
    {                                                                                                     \
        const auto inner_result = (x);                                                                    \
        if (inner_result != SPV_REFLECT_RESULT_SUCCESS)                                                   \
        {                                                                                                 \
            throw AIHoloImager::VulkanErrorException<SpvReflectResult>(inner_result, __FILE__, __LINE__); \
        }                                                                                                 \
    }
