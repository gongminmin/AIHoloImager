// Copyright (c) 2025 Minmin Gong
//

#pragma once

#define DEFINE_INTERNAL(ClassName)                        \
public:                                                   \
    ClassName##Internal& Internal() noexcept;             \
    const ClassName##Internal& Internal() const noexcept; \
                                                          \
private:
