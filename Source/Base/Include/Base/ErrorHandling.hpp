// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <cassert>
#include <stdexcept>
#include <string>
#include <string_view>

#include "Base/MiniWindows.hpp"

namespace AIHoloImager
{
    std::string CombineFileLine(std::string_view file, uint32_t line);
    std::string CombineFileLine(HRESULT hr, std::string_view file, uint32_t line);
    void Verify(bool value);

#ifdef _WIN32
    class HrException : public std::runtime_error
    {
    public:
        HrException(HRESULT hr, std::string_view file, uint32_t line)
            : std::runtime_error(CombineFileLine(hr, std::move(file), line)), hr_(hr)
        {
        }

        HRESULT Error() const noexcept
        {
            return hr_;
        }

    private:
        const HRESULT hr_;
    };
#endif

    [[noreturn]] inline void Unreachable([[maybe_unused]] std::string_view msg = {})
    {
#if defined(_MSC_VER)
        assert(false);
        __assume(false);
#else
        __builtin_unreachable();
#endif
    }
} // namespace AIHoloImager

#ifdef _WIN32
    #define TIFHR(x)                                                           \
        {                                                                      \
            const auto inner_hr = (x);                                         \
            if (FAILED(inner_hr))                                              \
            {                                                                  \
                throw AIHoloImager::HrException(inner_hr, __FILE__, __LINE__); \
            }                                                                  \
        }
#endif
