// Copyright (c) 2024 Minmin Gong
//

#pragma once

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif

#include <memory>

namespace AIHoloImager
{
#ifdef _WIN32
    class Win32HandleDeleter final
    {
    public:
        void operator()(HANDLE handle)
        {
            if (handle != INVALID_HANDLE_VALUE)
            {
                ::CloseHandle(handle);
            }
        }
    };
    using Win32UniqueHandle = std::unique_ptr<void, Win32HandleDeleter>;

    inline Win32UniqueHandle MakeWin32UniqueHandle(HANDLE handle)
    {
        return Win32UniqueHandle(handle);
    }
#endif
} // namespace AIHoloImager
