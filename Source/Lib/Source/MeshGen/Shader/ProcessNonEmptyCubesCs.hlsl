// Copyright (c) 2024 Minmin Gong
//

#include "MarchingCubesUtil.hlslh"

#define BLOCK_DIM 256

cbuffer param_cb : register(b0)
{
    uint size;
    uint total_cubes;
    float isovalue;
};

Buffer<uint> edge_table : register(t0);
Buffer<uint> triangle_table : register(t1);
Buffer<uint> non_empty_cube_offsets : register(t2);
Buffer<float> sdf : register(t3);

RWBuffer<uint> non_empty_cube_ids : register(u0);
RWBuffer<uint> non_empty_cube_indices : register(u1);
RWBuffer<uint2> non_empty_num_vertices_indices : register(u2);

[numthreads(BLOCK_DIM, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (dtid.x >= total_cubes)
    {
        return;
    }

    const uint cid = dtid.x;
    const uint offset = non_empty_cube_offsets[cid];
    [branch]
    if (offset == non_empty_cube_offsets[cid + 1])
    {
        return;
    }

    const uint OwnedEdges[] = {0, 3, 8};

    non_empty_cube_ids[offset] = cid;

    const uint3 coord = DecomposeCoord(cid, size);
    const uint cube_index = CalcCubeIndex(sdf, coord, size, isovalue);

    non_empty_cube_indices[offset] = cube_index;
    const uint cube_num_indices = triangle_table[cube_index * 16 + 0];

    const uint edges = edge_table[cube_index];
    uint cube_num_vertices = 0;
    if (edges != 0)
    {
        for (uint i = 0; i < sizeof(OwnedEdges) / sizeof(OwnedEdges[0]); ++i)
        {
            const uint e = OwnedEdges[i];
            if (edges & (1U << e))
            {
                ++cube_num_vertices;
            }
        }
    }

    non_empty_num_vertices_indices[offset] = uint2(cube_num_vertices, cube_num_indices);
}
