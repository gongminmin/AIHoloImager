// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>
#include <directx/d3d12video.h>

namespace AIHoloImager
{
    struct D3D12Traits
    {
        using DeviceType = ID3D12Device*;
        using CommandQueueType = ID3D12CommandQueue*;

        using CommandListType = ID3D12CommandList*;
        using GraphicsCommandListType = ID3D12GraphicsCommandList*;
        using ComputeCommandListType = ID3D12GraphicsCommandList*;
        using VideoEncodeCommandListType = ID3D12VideoEncodeCommandList*;

        using ResourceType = ID3D12Resource*;
        using BufferType = ID3D12Resource*;
        using TextureType = ID3D12Resource*;

        using SharedHandleType = HANDLE;
    };
} // namespace AIHoloImager
