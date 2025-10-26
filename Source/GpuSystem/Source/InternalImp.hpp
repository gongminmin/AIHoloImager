// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <cassert>

#define IMP_INTERNAL(ClassName)                                     \
    ClassName##Internal& ClassName::Internal() noexcept             \
    {                                                               \
        assert(impl_);                                              \
        return *impl_;                                              \
    }                                                               \
                                                                    \
    const ClassName##Internal& ClassName::Internal() const noexcept \
    {                                                               \
        return const_cast<ClassName&>(*this).Internal();            \
    }


#define IMP_INTERNAL2(ClassName, ReturnName)                         \
    ReturnName##Internal& ClassName::Internal() noexcept             \
    {                                                                \
        assert(impl_);                                               \
        return *impl_;                                               \
    }                                                                \
                                                                     \
    const ReturnName##Internal& ClassName::Internal() const noexcept \
    {                                                                \
        return const_cast<ClassName&>(*this).Internal();             \
    }

#define EMPTY_IMP(ClassName)                           \
    class ClassName::Impl : public ClassName##Internal \
    {                                                  \
    };
