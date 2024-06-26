// Copyright (c) 2024 Minmin Gong
//

#pragma once

#define DISALLOW_COPY_AND_ASSIGN(ClassName)     \
    ClassName(ClassName const& other) = delete; \
    ClassName& operator=(ClassName const& other) = delete;

#define DISALLOW_COPY_MOVE_AND_ASSIGN(ClassName) \
    ClassName(ClassName&& other) = delete;       \
    ClassName& operator=(ClassName&& other) = delete;
