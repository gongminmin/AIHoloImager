// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <cstdint>
#include <type_traits>

#include <directx/d3d12.h>

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
    constexpr bool EnumHasAny(T e, T v)
    {
        return (e & v) != static_cast<T>(0);
    }
    template <typename T>
    constexpr bool EnumHasAll(T e, T v)
    {
        return (e & v) == v;
    }
    template <typename T>
    constexpr bool EnumHasNone(T e, T v)
    {
        return (e & v) == static_cast<T>(0);
    }

    enum class GpuHeap
    {
        Default,
        Upload,
        ReadBack,
    };

    D3D12_HEAP_TYPE ToD3D12HeapType(GpuHeap heap);

    enum class GpuResourceFlag : uint32_t
    {
        None = 0,
        RenderTarget = 1U << 0,
        DepthStencil = 1U << 1,
        UnorderedAccess = 1U << 2,
    };
    ENUM_CLASS_BITWISE_OPERATORS(GpuResourceFlag);

    D3D12_RESOURCE_FLAGS ToD3D12ResourceFlags(GpuResourceFlag flags);

    enum class GpuResourceState
    {
        Common,

        ColorWrite,
        DepthWrite,

        UnorderedAccess,

        CopySrc,
        CopyDst,

        RayTracingAS,
    };
    D3D12_RESOURCE_STATES ToD3D12ResourceState(GpuResourceState state);
} // namespace AIHoloImager
