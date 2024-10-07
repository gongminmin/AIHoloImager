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
Buffer<float> sdf : register(t1);

RWBuffer<uint> non_empty_cube_flags : register(u0);

[numthreads(BLOCK_DIM, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (dtid.x >= total_cubes)
    {
        return;
    }

    const uint cid = dtid.x;
    const uint3 coord = DecomposeCoord(cid, size);
    const uint cube_index = CalcCubeIndex(sdf, coord, size, isovalue);
    non_empty_cube_flags[cid] = (edge_table[cube_index] != 0);
}
