// Copyright (c) 2024 Minmin Gong
//

#include "MarchingCubesUtil.hlslh"

static const uint32_t BlockDim = 256;
static const uint32_t MaxGroupDim = 65535;

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

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    const uint32_t cid = dtid.y * MaxGroupDim + dtid.x;

    [branch]
    if (cid >= total_cubes)
    {
        return;
    }

    const uint32_t offset = cube_offsets[cid];
    [branch]
    if (offset == ~0U)
    {
        return;
    }

    const uint32_t OwnedEdges[] = {0, 3, 8};

    non_empty_cube_ids[offset] = cid;

    const uint32_t3 coord = DecomposeCoord(cid, size);
    const uint32_t cube_index = CalcCubeIndex(scalar_deformation, coord, size, isovalue);

    non_empty_cube_indices[offset] = cube_index;
    const uint32_t cube_num_indices = triangle_table[cube_index * 16 + 0];

    const uint32_t edges = edge_table[cube_index];
    uint32_t cube_num_vertices = 0;
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

    uint32_t vertex_addr;
    InterlockedAdd(counter[1], cube_num_vertices, vertex_addr);

    uint32_t index_addr;
    InterlockedAdd(counter[2], cube_num_indices, index_addr);

    vertex_index_offsets[offset] = uint32_t2(vertex_addr, index_addr);
}
