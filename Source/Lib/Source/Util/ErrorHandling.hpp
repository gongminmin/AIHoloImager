// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif

#define AI_HOLO_IMAGER_UNREACHABLE(msg) __assume(false)

namespace AIHoloImager
{
    std::string CombineFileLine(std::string_view file, uint32_t line);
    std::string CombineFileLine(HRESULT hr, std::string_view file, uint32_t line);
    void Verify(bool value);

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
        HRESULT const hr_;
    };

    [[noreturn]] inline void Unreachable()
    {
#if defined(_MSC_VER)
        __assume(false);
#else
        __builtin_unreachable();
#endif
    }
} // namespace AIHoloImager


#define TIFHR(hr)                                                    \
    {                                                                \
        if (FAILED(hr))                                              \
        {                                                            \
            throw AIHoloImager::HrException(hr, __FILE__, __LINE__); \
        }                                                            \
    }
