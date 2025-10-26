// Copyright (c) 2025 Minmin Gong
//

#pragma once

#define D3D12_DEFINE_IMP(ClassName)                  \
    D3D12##ClassName& D3D12Imp(Gpu##ClassName& var); \
    const D3D12##ClassName& D3D12Imp(const Gpu##ClassName& var);

#define D3D12_IMP_IMP(ClassName)                                     \
    D3D12##ClassName& D3D12Imp(Gpu##ClassName& var)                  \
    {                                                                \
        return static_cast<D3D12##ClassName&>(var.Internal());       \
    }                                                                \
                                                                     \
    const D3D12##ClassName& D3D12Imp(const Gpu##ClassName& var)      \
    {                                                                \
        return static_cast<const D3D12##ClassName&>(var.Internal()); \
    }
