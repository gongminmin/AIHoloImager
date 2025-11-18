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

Buffer<uint16_t> edge_table;
Texture3D<float4> scalar_deformation;

RWBuffer<uint32_t> cube_offsets;
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
