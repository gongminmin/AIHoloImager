// Copyright (c) 2025 Minmin Gong
//

#include <type_traits>

namespace AIHoloImager
{
#define ENUM_CLASS_BITWISE_OPERATORS(Enum)                                                     \
    constexpr Enum operator|(Enum lhs, Enum rhs)                                               \
    {                                                                                          \
        using Underlying = typename std::underlying_type_t<Enum>;                              \
        return static_cast<Enum>(static_cast<Underlying>(lhs) | static_cast<Underlying>(rhs)); \
    }                                                                                          \
    constexpr Enum operator&(Enum lhs, Enum rhs)                                               \
    {                                                                                          \
        using Underlying = typename std::underlying_type_t<Enum>;                              \
        return static_cast<Enum>(static_cast<Underlying>(lhs) & static_cast<Underlying>(rhs)); \
    }                                                                                          \
    constexpr Enum operator^(Enum lhs, Enum rhs)                                               \
    {                                                                                          \
        using Underlying = typename std::underlying_type_t<Enum>;                              \
        return static_cast<Enum>(static_cast<Underlying>(lhs) ^ static_cast<Underlying>(rhs)); \
    }                                                                                          \
    constexpr Enum operator~(Enum rhs)                                                         \
    {                                                                                          \
        using Underlying = typename std::underlying_type_t<Enum>;                              \
        return static_cast<Enum>(~static_cast<Underlying>(rhs));                               \
    }                                                                                          \
    constexpr Enum& operator|=(Enum& lhs, Enum rhs)                                            \
    {                                                                                          \
        lhs = lhs | rhs;                                                                       \
        return lhs;                                                                            \
    }                                                                                          \
    constexpr Enum& operator&=(Enum& lhs, Enum rhs)                                            \
    {                                                                                          \
        lhs = lhs & rhs;                                                                       \
        return lhs;                                                                            \
    }                                                                                          \
    constexpr Enum& operator^=(Enum& lhs, Enum rhs)                                            \
    {                                                                                          \
        lhs = lhs ^ rhs;                                                                       \
        return lhs;                                                                            \
    }

    template <typename T>
    constexpr bool EnumHasAny(T e, T v) noexcept
    {
        return (e & v) != static_cast<T>(0);
    }
    template <typename T>
    constexpr bool EnumHasAll(T e, T v) noexcept
    {
        return (e & v) == v;
    }
    template <typename T>
    constexpr bool EnumHasNone(T e, T v) noexcept
    {
        return (e & v) == static_cast<T>(0);
    }
} // namespace AIHoloImager
