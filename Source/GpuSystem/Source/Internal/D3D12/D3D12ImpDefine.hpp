// Copyright (c) 2025-2026 Minmin Gong
//

#pragma once

#define D3D12_DEFINE_IMP(ClassName)                              \
    D3D12##ClassName& D3D12Imp(Gpu##ClassName& var);             \
    const D3D12##ClassName& D3D12Imp(const Gpu##ClassName& var); \
    D3D12##ClassName& D3D12Imp(Gpu##ClassName##Internal& var);   \
    const D3D12##ClassName& D3D12Imp(const Gpu##ClassName##Internal& var);

#define D3D12_IMP_IMP(ClassName)                                          \
    D3D12##ClassName& D3D12Imp(Gpu##ClassName& var)                       \
    {                                                                     \
        return static_cast<D3D12##ClassName&>(var.Internal());            \
    }                                                                     \
    const D3D12##ClassName& D3D12Imp(const Gpu##ClassName& var)           \
    {                                                                     \
        return static_cast<const D3D12##ClassName&>(var.Internal());      \
    }                                                                     \
    D3D12##ClassName& D3D12Imp(Gpu##ClassName##Internal& var)             \
    {                                                                     \
        return static_cast<D3D12##ClassName&>(var);                       \
    }                                                                     \
    const D3D12##ClassName& D3D12Imp(const Gpu##ClassName##Internal& var) \
    {                                                                     \
        return static_cast<const D3D12##ClassName&>(var);                 \
    }
