// Copyright (c) 2025 Minmin Gong
//

#include "Base/Dll.hpp"

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

namespace AIHoloImager
{
    class Dll::Impl
    {
    public:
        explicit Impl(const std::filesystem::path& path)
        {
#ifdef _WIN32
            native_handle_ = ::LoadLibraryExA(path.string().c_str(), nullptr, 0);
#else
            native_handle_ = ::dlopen(path.string().c_str(), RTLD_LAZY);
#endif
        }

        ~Impl() noexcept
        {
            this->Free();
        }

        Impl(Impl&& other) noexcept : native_handle_(std::exchange(other.native_handle_, nullptr))
        {
        }

        Impl& operator=(Impl&& other) noexcept
        {
            if (this != &other)
            {
                if (native_handle_ != nullptr)
                {
                    this->Free();
                }

                native_handle_ = std::exchange(other.native_handle_, nullptr);
            }

            return *this;
        }

        explicit operator bool() const noexcept
        {
            return native_handle_ != nullptr;
        }

        void* Func(const char* proc_name) const noexcept
        {
#ifdef _WIN32
            return reinterpret_cast<void*>(::GetProcAddress(native_handle_, proc_name));
#else
            return ::dlsym(native_handle_, proc_name);
#endif
        }

    private:
        void Free() noexcept
        {
            if (native_handle_ != nullptr)
            {
#ifdef _WIN32
                ::FreeLibrary(native_handle_);
#else
                ::dlclose(native_handle_);
#endif

                native_handle_ = nullptr;
            }
        }

    private:
#ifdef _WIN32
        HMODULE native_handle_;
#else
        void* native_handle_;
#endif
    };

    Dll::Dll() noexcept = default;
    Dll::Dll(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path))
    {
    }
    Dll::Dll(Dll&& other) noexcept = default;
    Dll::~Dll() noexcept = default;

    Dll& Dll::operator=(Dll&& other) noexcept = default;

    Dll::operator bool() const noexcept
    {
        return static_cast<bool>(*impl_);
    }

    void* Dll::Func(const char* proc_name) const noexcept
    {
        return impl_->Func(proc_name);
    }
} // namespace AIHoloImager
