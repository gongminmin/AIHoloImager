// Copyright (c) 2025 Minmin Gong
//

#include "Common.hlslh"

static const uint32_t BlockDim = 256;
static const uint32_t MaxProbeTime = 1024;
static const uint32_t Empty = ~0U;

cbuffer param_cb
{
    uint32_t num_coords;
    uint32_t hash_size;
};

Buffer<uint32_t4> coords_buff;

RWBuffer<uint32_t> hash_table;

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (dtid.x >= num_coords)
    {
        return;
    }

    const uint32_t4 coord = coords_buff[dtid.x]; // (B, z, y, x)

    uint32_t slot = HashFunc(coord) % hash_size;

    [allow_uav_condition]
    for (uint32_t t = 0; t < MaxProbeTime; ++t)
    {
        uint32_t ori_val;
        InterlockedCompareExchange(hash_table[slot * 5 + 0], Empty, coord.x, ori_val);
        if (ori_val == Empty)
        {
            hash_table[slot * 5 + 1] = coord.y;
            hash_table[slot * 5 + 2] = coord.z;
            hash_table[slot * 5 + 3] = coord.w;
            hash_table[slot * 5 + 4] = dtid.x;
            break;
        }

        slot = (slot + 1) % hash_size;
    }
}
