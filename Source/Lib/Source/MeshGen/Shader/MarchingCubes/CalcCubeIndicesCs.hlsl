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
Texture3D<float4> scalar_deformation;

RWBuffer<uint32_t> cube_offsets;
RWBuffer<uint32_t> counter;

groupshared uint32_t group_cube_offset;

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID, uint32_t group_index : SV_GroupIndex)
{
    uint32_t non_empty = 0;

    const uint32_t cid = dtid.y * MaxGroupDim + dtid.x;
    [branch]
    if (cid < total_cubes)
    {
        const uint32_t3 coord = DecomposeCoord(cid, size);
        const uint32_t cube_index = CalcCubeIndex(scalar_deformation, coord, size, isovalue);

        non_empty = (edge_table[cube_index] != 0);
    }

    const uint32_t cube_offset_in_group = ScanBlock(group_index, non_empty);
    if (group_index == BlockDim - 1)
    {
        InterlockedAdd(counter[0], cube_offset_in_group + non_empty, group_cube_offset);
    }
    GroupMemoryBarrierWithGroupSync();

    if (cid < total_cubes)
    {
        cube_offsets[cid] = non_empty ? group_cube_offset + cube_offset_in_group : ~0U;
    }
}
