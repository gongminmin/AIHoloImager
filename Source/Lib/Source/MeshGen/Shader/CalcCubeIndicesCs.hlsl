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
Buffer<float4> sdf_deformation : register(t1);

RWBuffer<uint> cube_offsets : register(u0);
RWBuffer<uint> counter : register(u1);

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
    const uint cube_index = CalcCubeIndex(sdf_deformation, coord, size, isovalue);

    if (edge_table[cube_index] != 0)
    {
        uint addr;
        InterlockedAdd(counter[0], 1, addr);
        cube_offsets[cid] = addr;
    }
    else
    {
        cube_offsets[cid] = ~0U;
    }
}
