// Copyright (c) 2024 Minmin Gong
//

#include "MarchingCubesUtil.hlslh"

#define BLOCK_DIM 256

cbuffer param_cb : register(b0)
{
    uint32_t size;
    uint32_t total_cubes;
    float isovalue;
};

Buffer<uint16_t> edge_table : register(t0);
Texture3D<float4> scalar_deformation : register(t1);

RWBuffer<uint32_t> cube_offsets : register(u0);
RWBuffer<uint32_t> counter : register(u1);

[numthreads(BLOCK_DIM, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (dtid.x >= total_cubes)
    {
        return;
    }

    const uint32_t cid = dtid.x;
    const uint32_t3 coord = DecomposeCoord(cid, size);
    const uint32_t cube_index = CalcCubeIndex(scalar_deformation, coord, size, isovalue);

    if (edge_table[cube_index] != 0)
    {
        uint32_t addr;
        InterlockedAdd(counter[0], 1, addr);
        cube_offsets[cid] = addr;
    }
    else
    {
        cube_offsets[cid] = ~0U;
    }
}
