// Copyright (c) 2025 Minmin Gong
//

#include "Common.hlslh"

static const uint32_t BlockDim = 256;
static const uint32_t MaxProbeTime = 1024;
static const uint32_t Empty = ~0U;

cbuffer param_cb
{
    uint32_t num_offsets;
    uint32_t num_coords;
    uint32_t hash_size;
};

Buffer<uint32_t4> coords_buff;
Buffer<uint32_t> hash_table;
Buffer<int32_t3> offsets_buff;

RWBuffer<uint32_t2> nei_indices;
RWBuffer<uint32_t> nei_indices_count;

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (dtid.x >= num_coords)
    {
        return;
    }

    const uint32_t4 coord = asuint(asint(coords_buff[dtid.x]) + int32_t4(0, offsets_buff[dtid.y])); // (B, z, y, x)

    uint32_t slot = HashFunc(coord) % hash_size;

    for (uint32_t t = 0; t < MaxProbeTime; ++t)
    {
        if (hash_table[slot * 5 + 0] == Empty)
        {
            break;
        }
        else if ((hash_table[slot * 5 + 0] == coord.x) && (hash_table[slot * 5 + 1] == coord.y) && (hash_table[slot * 5 + 2] == coord.z) && (hash_table[slot * 5 + 3] == coord.w))
        {
            uint32_t addr;
            InterlockedAdd(nei_indices_count[dtid.y], 1, addr);
            nei_indices[dtid.y * num_coords + addr] = uint32_t2(dtid.x, hash_table[slot * 5 + 4]);
            break;
        }

        slot = (slot + 1) % hash_size;
    }
}
