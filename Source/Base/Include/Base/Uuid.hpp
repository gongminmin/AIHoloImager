// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <array>
#include <cstdint>

#ifdef _WIN32
    #include <guiddef.h>
#endif

namespace AIHoloImager
{
    struct Uuid
    {
        uint32_t data1;
        uint16_t data2;
        uint16_t data3;
        uint8_t data4[8];

        Uuid() noexcept = default;
        constexpr Uuid(uint32_t d1, uint16_t d2, uint16_t d3, const std::array<uint8_t, 8> d4) noexcept
            : data1(d1), data2(d2), data3(d3), data4{d4[0], d4[1], d4[2], d4[3], d4[4], d4[5], d4[6], d4[7]}
        {
        }

#ifdef _WIN32
        constexpr Uuid(const GUID& value) noexcept
            : Uuid(value.Data1, value.Data2, value.Data3,
                  {value.Data4[0], value.Data4[1], value.Data4[2], value.Data4[3], value.Data4[4], value.Data4[5], value.Data4[6],
                      value.Data4[7]})
        {
        }

        operator const GUID&() const noexcept
        {
            return reinterpret_cast<const GUID&>(*this);
        }
#endif
    };

    template <typename T>
    const Uuid& UuidOf();

    template <typename T>
    inline const Uuid& UuidOf([[maybe_unused]] T* p)
    {
        return UuidOf<T>();
    }

    template <typename T>
    inline const Uuid& UuidOf([[maybe_unused]] const T* p)
    {
        return UuidOf<T>();
    }

    template <typename T>
    inline const Uuid& UuidOf([[maybe_unused]] T& p)
    {
        return UuidOf<T>();
    }

    template <typename T>
    inline const Uuid& UuidOf([[maybe_unused]] const T& p)
    {
        return UuidOf<T>();
    }
} // namespace AIHoloImager

#define DEFINE_UUID_OF(x)                                            \
    template <>                                                      \
    const AIHoloImager::Uuid& AIHoloImager::UuidOf<x>()              \
    {                                                                \
        return reinterpret_cast<const AIHoloImager::Uuid&>(IID_##x); \
    }
