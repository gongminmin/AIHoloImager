// Copyright (c) 2024 Minmin Gong
//

#include "MarchingCubesUtil.hlslh"

static const uint32_t BlockDim = 256;

cbuffer param_cb : register(b0)
{
    uint32_t size;
    uint32_t total_cubes;
    float isovalue;
};

Buffer<uint16_t> edge_table : register(t0);
Buffer<uint16_t> triangle_table : register(t1);
Buffer<uint32_t> cube_offsets : register(t2);
Texture3D<float4> scalar_deformation : register(t3);

RWBuffer<uint32_t> non_empty_cube_ids : register(u0);
RWBuffer<uint32_t> non_empty_cube_indices : register(u1);
RWBuffer<uint32_t2> vertex_index_offsets : register(u2);
RWBuffer<uint32_t> counter : register(u3);

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (dtid.x >= total_cubes)
    {
        return;
    }

    const uint32_t cid = dtid.x;
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
