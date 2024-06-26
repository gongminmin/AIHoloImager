// Copyright (c) 2024 Minmin Gong
//

#include "ErrorHandling.hpp"

#include <iomanip>
#include <sstream>

namespace AIHoloImager
{
    std::string CombineFileLine(std::string_view file, uint32_t line)
    {
        std::ostringstream ss;
        ss << " from " << file << ": " << line;
        return ss.str();
    }

    std::string CombineFileLine(HRESULT hr, std::string_view file, uint32_t line)
    {
        std::ostringstream ss;
        ss << "HRESULT of 0x" << std::hex << std::setfill('0') << std::setw(8) << static_cast<uint32_t>(hr);
        ss << CombineFileLine(std::move(file), line);
        return ss.str();
    }

    void Verify(bool x)
    {
        if (!x)
        {
            throw std::runtime_error("Verify failed.");
        }
    }
} // namespace AIHoloImager
