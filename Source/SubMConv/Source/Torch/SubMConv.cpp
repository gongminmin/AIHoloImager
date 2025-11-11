// Copyright (c) 2025 Minmin Gong
//

#include "SubMConv.hpp"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuConstantBuffer.hpp"

#include "CompiledShader/Torch/Dxil/BuildCoordHashCs.h"
#include "CompiledShader/Torch/Dxil/FindAvailableNeighborsCs.h"

namespace AIHoloImager
{
    SubMConv3DHelper::SubMConv3DHelper(size_t gpu_system, torch::Device torch_device)
        : gpu_system_(*reinterpret_cast<GpuSystem*>(gpu_system)), torch_device_(std::move(torch_device)),
          tensor_converter_(gpu_system_, torch_device_)
    {
        {
            const ShaderInfo shader = {DEFINE_SHADER(BuildCoordHashCs)};
            build_coord_hash_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {DEFINE_SHADER(FindAvailableNeighborsCs)};
            find_available_neighbors_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
    }

    SubMConv3DHelper::~SubMConv3DHelper() = default;

    void SubMConv3DHelper::BuildCoordsMap(torch::Tensor coords)
    {
        num_coords_ = static_cast<uint32_t>(coords.size(0));

        const uint32_t expected_hash_buff_size = num_coords_ * 4 * sizeof(uint32_t) * 5;
        if (coord_hash_.Size() < expected_hash_buff_size)
        {
            coord_hash_ = GpuBuffer(
                gpu_system_, expected_hash_buff_size, GpuHeap::Default, GpuResourceFlag::UnorderedAccess, "SubMConv3DHelper.coord_hash_");
            coord_hash_srv_ = GpuShaderResourceView(gpu_system_, coord_hash_, GpuFormat::R32_Uint);
            coord_hash_uav_ = GpuUnorderedAccessView(gpu_system_, coord_hash_, GpuFormat::R32_Uint);
        }

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        tensor_converter_.Convert(
            cmd_list, coords.to(torch::kInt32), coords_buff_, GpuHeap::Default, GpuResourceFlag::None, "SubMConv3DHelper.coords_buff_");
        coords_srv_ = GpuShaderResourceView(gpu_system_, coords_buff_, GpuFormat::RGBA32_Uint);

        {
            constexpr uint32_t BlockDim = 256;

            GpuConstantBufferOfType<BuildCoordHashConstantBuffer> build_coord_hash_cb(gpu_system_, "build_coord_hash_cb");
            build_coord_hash_cb->num_coords = num_coords_;
            build_coord_hash_cb->hash_size = coord_hash_.Size() / (sizeof(uint32_t) * 5);
            build_coord_hash_cb.UploadStaging();

            {
                const uint32_t clear_clr[] = {~0U, ~0U, ~0U, ~0U};
                cmd_list.Clear(coord_hash_uav_, clear_clr);
            }

            const GpuConstantBuffer* cbs[] = {&build_coord_hash_cb};
            const GpuShaderResourceView* srvs[] = {&coords_srv_};
            GpuUnorderedAccessView* uavs[] = {&coord_hash_uav_};
            const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
            cmd_list.Compute(build_coord_hash_pipeline_, DivUp(num_coords_, BlockDim), 1, 1, shader_binding);
        }

        gpu_system_.Execute(std::move(cmd_list));
    }

    std::array<torch::Tensor, 2> SubMConv3DHelper::FindAvailableNeighbors(
        const std::array<int32_t, 3>& base, const std::vector<std::array<uint32_t, 3>>& offsets)
    {
        const uint32_t num_offsets = static_cast<uint32_t>(offsets.size());

        const uint32_t expected_indices_size = num_offsets * num_coords_ * sizeof(glm::uvec2);
        if (nei_indices_.Size() < expected_indices_size)
        {
            nei_indices_ = GpuBuffer(gpu_system_, expected_indices_size, GpuHeap::Default,
                GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable, "SubMConv3DHelper.nei_indices_");
            nei_indices_uav_ = GpuUnorderedAccessView(gpu_system_, nei_indices_, GpuFormat::RG32_Uint);
        }

        const uint32_t expected_index_count_size = num_offsets * sizeof(uint32_t);
        if (nei_indices_count_.Size() < expected_index_count_size)
        {
            nei_indices_count_ = GpuBuffer(gpu_system_, expected_index_count_size, GpuHeap::Default,
                GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable, "SubMConv3DHelper.nei_indices_count_");
            nei_indices_count_uav_ = GpuUnorderedAccessView(gpu_system_, nei_indices_count_, GpuFormat::R32_Uint);
        }

        const uint32_t expected_offsets_size = num_offsets * sizeof(glm::ivec3);
        if (offsets_buff_.Size() < expected_offsets_size)
        {
            offsets_buff_ =
                GpuBuffer(gpu_system_, expected_offsets_size, GpuHeap::Default, GpuResourceFlag::None, "SubMConv3DHelper.offsets_buff_");
            offsets_srv_ = GpuShaderResourceView(gpu_system_, offsets_buff_, GpuFormat::RGB32_Sint);
        }

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        cmd_list.Upload(offsets_buff_, [num_offsets, &base, &offsets](void* dst_data) {
            int32_t* i32_dst_data = static_cast<int32_t*>(dst_data);
            for (uint32_t i = 0; i < num_offsets; ++i)
            {
                i32_dst_data[i * 3 + 0] = base[0] + offsets[i][0];
                i32_dst_data[i * 3 + 1] = base[1] + offsets[i][1];
                i32_dst_data[i * 3 + 2] = base[2] + offsets[i][2];
            }
        });

        {
            constexpr uint32_t BlockDim = 256;

            GpuConstantBufferOfType<FindAvailableNeighborsConstantBuffer> find_avail_nei_cb(gpu_system_, "find_avail_nei_cb");
            find_avail_nei_cb->num_offsets = num_offsets;
            find_avail_nei_cb->num_coords = num_coords_;
            find_avail_nei_cb->hash_size = coord_hash_.Size() / (sizeof(uint32_t) * 5);
            find_avail_nei_cb.UploadStaging();

            {
                const uint32_t clear_clr[] = {0, 0, 0, 0};
                cmd_list.Clear(nei_indices_count_uav_, clear_clr);
            }

            const GpuConstantBuffer* cbs[] = {&find_avail_nei_cb};
            const GpuShaderResourceView* srvs[] = {&coords_srv_, &coord_hash_srv_, &offsets_srv_};
            GpuUnorderedAccessView* uavs[] = {&nei_indices_uav_, &nei_indices_count_uav_};
            const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
            cmd_list.Compute(find_available_neighbors_pipeline_, DivUp(num_coords_, BlockDim), num_offsets, 1, shader_binding);
        }

        torch::Tensor nei_indices = tensor_converter_.Convert(
            cmd_list, nei_indices_, {static_cast<int32_t>(num_offsets), static_cast<int32_t>(num_coords_), 2}, torch::kInt32);
        torch::Tensor nei_indices_size =
            tensor_converter_.Convert(cmd_list, nei_indices_count_, {static_cast<int32_t>(num_offsets)}, torch::kInt32);

        gpu_system_.Execute(std::move(cmd_list));

        return {std::move(nei_indices), std::move(nei_indices_size)};
    }
} // namespace AIHoloImager
