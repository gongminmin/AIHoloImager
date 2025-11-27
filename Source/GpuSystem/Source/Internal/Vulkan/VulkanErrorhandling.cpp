// Copyright (c) 2025 Minmin Gong
//

#include "VulkanErrorHandling.hpp"

#include <iomanip>
#include <sstream>

namespace AIHoloImager
{
    template <typename ResultType>
    std::string CombineFileLine(ResultType result, std::string_view file, uint32_t line)
    {
        std::ostringstream ss;
        if constexpr (std::is_same_v<ResultType, VkResult>)
        {
            ss << "VkResult";
        }
        else
        {
            static_assert(std::is_same_v<ResultType, SpvReflectResult>);
            ss << "SpvReflectResult";
        }
        ss << " of 0x" << std::hex << std::setfill('0') << std::setw(8) << static_cast<uint32_t>(result);
        ss << CombineFileLine(std::move(file), line);
        return ss.str();
    }
} // namespace AIHoloImager
