// Copyright (c) 2024 Minmin Gong
//

#include "MarchingCubesUtil.hlslh"

static const uint32_t BlockDim = 256;
static const uint32_t MaxGroupDim = 65535;

#define DATA_TYPE uint32_t
#include "../PrefixSumScanner/PrefixSumBlock.hlslh"

cbuffer param_cb
{
    uint32_t size;
    uint32_t total_cubes;
    float isovalue;
};

#ifdef __spirv__
[[vk::image_format("r16ui")]]
#endif
Buffer<uint32_t> edge_table;
#ifdef __spirv__
[[vk::image_format("r16ui")]]
#endif
Buffer<uint32_t> triangle_table;
Buffer<uint32_t> cube_offsets;
Texture3D<float4> scalar_deformation;

RWBuffer<uint32_t> non_empty_cube_ids;
RWBuffer<uint32_t> non_empty_cube_indices;
RWBuffer<uint32_t2> vertex_index_offsets;
RWBuffer<uint32_t> counter;

groupshared uint32_t group_vertex_offset;
groupshared uint32_t group_index_offset;

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID, uint32_t group_index : SV_GroupIndex)
{
    uint32_t offset = 0;
    uint32_t cube_num_vertices = 0;
    uint32_t cube_num_indices = 0;

    const uint32_t cid = dtid.y * MaxGroupDim + dtid.x;
    [branch]
    if (cid < total_cubes)
    {
        offset = cube_offsets[cid];
        [branch]
        if (offset != ~0U)
        {
            const uint32_t OwnedEdges[] = {0, 3, 8};

            non_empty_cube_ids[offset] = cid;

            const uint32_t3 coord = DecomposeCoord(cid, size);
            const uint32_t cube_index = CalcCubeIndex(scalar_deformation, coord, size, isovalue);

            non_empty_cube_indices[offset] = cube_index;
            cube_num_indices = triangle_table[cube_index * 16 + 0];

            const uint32_t edges = edge_table[cube_index];
            if (edges != 0)
            {
                for (uint32_t i = 0; i < sizeof(OwnedEdges) / sizeof(OwnedEdges[0]); ++i)
                {
                    const uint32_t e = OwnedEdges[i];
                    if (edges & (1U << e))
                    {
                        ++cube_num_vertices;
                    }
                }
            }
        }
    }

    const uint32_t vertex_offset_in_group = ScanBlock(group_index, cube_num_vertices);
    const uint32_t index_offset_in_group = ScanBlock(group_index, cube_num_indices);
    if (group_index == BlockDim - 1)
    {
        InterlockedAdd(counter[1], vertex_offset_in_group + cube_num_vertices, group_vertex_offset);
        InterlockedAdd(counter[2], index_offset_in_group + cube_num_indices, group_index_offset);
    }
    GroupMemoryBarrierWithGroupSync();

    if ((cube_num_vertices > 0) || (cube_num_indices > 0))
    {
        vertex_index_offsets[offset] = uint32_t2(group_vertex_offset + vertex_offset_in_group, group_index_offset + index_offset_in_group);
    }
}
