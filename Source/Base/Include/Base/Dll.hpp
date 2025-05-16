// Copyright (c) 2025 Minmin Gong
//

#include <filesystem>
#include <memory>

#include "Base/Noncopyable.hpp"

namespace AIHoloImager
{
    class Dll final
    {
        DISALLOW_COPY_AND_ASSIGN(Dll);

    public:
        Dll() noexcept;
        explicit Dll(const std::filesystem::path& path);
        Dll(Dll&& other) noexcept;
        ~Dll() noexcept;

        Dll& operator=(Dll&& other) noexcept;

        explicit operator bool() const noexcept;

        void* Func(const char* proc_name) const noexcept;

        template <typename T>
        T Func(const char* proc_name) const noexcept
        {
            return reinterpret_cast<T>(this->Func(proc_name));
        }

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
