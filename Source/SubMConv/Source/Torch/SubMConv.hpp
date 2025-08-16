// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <array>
#include <vector>

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4100) // Ignore unreferenced formal parameters
    #pragma warning(disable : 4127) // Ignore constant conditional expression
    #pragma warning(disable : 4244) // Ignore type conversion from `int` to `float`
    #pragma warning(disable : 4251) // Ignore non dll-interface as member
    #pragma warning(disable : 4267) // Ignore type conversion from `size_t` to something else
    #pragma warning(disable : 4324) // Ignore padded structure
    #pragma warning(disable : 4275) // Ignore non dll-interface base class
#endif
#include <torch/types.h>
#ifdef _MSC_VER
    #pragma warning(pop)
#endif

#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuShader.hpp"
#include "Gpu/GpuSystem.hpp"
#include "TensorConverter/TensorConverter.hpp"

namespace AIHoloImager
{
    class SubMConv3DHelper
    {
    public:
        SubMConv3DHelper(size_t gpu_system, torch::Device torch_device);
        ~SubMConv3DHelper();

        void BuildCoordsMap(torch::Tensor coords);
        std::array<torch::Tensor, 2> FindAvailableNeighbors(
            const std::array<int32_t, 3>& base, const std::vector<std::array<uint32_t, 3>>& offsets);

    private:
        GpuSystem& gpu_system_;
        torch::Device torch_device_;

        TensorConverter tensor_converter_;

        GpuBuffer coords_buff_;
        GpuShaderResourceView coords_srv_;
        uint32_t num_coords_ = 0;

        GpuBuffer coord_hash_;
        GpuShaderResourceView coord_hash_srv_;
        GpuUnorderedAccessView coord_hash_uav_;

        GpuBuffer nei_indices_;
        GpuUnorderedAccessView nei_indices_uav_;

        GpuBuffer nei_indices_count_;
        GpuUnorderedAccessView nei_indices_count_uav_;

        GpuBuffer offsets_buff_;
        GpuShaderResourceView offsets_srv_;

        struct BuildCoordHashConstantBuffer
        {
            uint32_t num_coords;
            uint32_t hash_size;
            uint32_t padding[2];
        };
        GpuComputePipeline build_coord_hash_pipeline_;

        struct FindAvailableNeighborsConstantBuffer
        {
            uint32_t num_offsets;
            uint32_t num_coords;
            uint32_t hash_size;
            uint32_t padding[1];
        };
        GpuComputePipeline find_available_neighbors_pipeline_;
    };
} // namespace AIHoloImager