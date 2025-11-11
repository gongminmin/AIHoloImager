// Copyright (c) 2025 Minmin Gong
//

#include "Common.hlslh"

static const uint32_t BlockDim = 256;
static const uint32_t MaxProbeTime = 256;

cbuffer param_cb
{
    uint32_t num_indices;
    uint32_t hash_size;
};

Buffer<uint32_t> indices_buff;
Buffer<uint32_t> hash_table;

RWBuffer<uint32_t> oppo_vert;

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (dtid.x >= num_indices)
    {
        return;
    }

    const uint32_t face_id = dtid.x / 3;
    const uint32_t index = dtid.x - face_id * 3;

    uint32_t2 edge = uint32_t2(indices_buff[face_id * 3 + ((index + 1) % 3)], indices_buff[face_id * 3 + ((index + 2) % 3)]);
    if (edge.y < edge.x)
    {
        Swap(edge.x, edge.y);
    }

    uint32_t slot = HashFunc(edge) % hash_size;
    const uint32_t this_vertex = indices_buff[face_id * 3 + index];

    bool found = false;
    for (uint32_t t = 0; t < MaxProbeTime; ++t)
    {
        if (hash_table[slot * 3 + 0] == ~0U)
        {
            oppo_vert[dtid.x] = ~0U;
            found = true;
        }
        else if ((hash_table[slot * 3 + 0] == edge.x) && (hash_table[slot * 3 + 1] == edge.y) && (hash_table[slot * 3 + 2] != this_vertex))
        {
            oppo_vert[dtid.x] = hash_table[slot * 3 + 2];
            found = true;
        }

        if (found)
        {
            break;
        }
        else
        {
            slot = (slot + 1) % hash_size;
        }
    }

    if (!found)
    {
        oppo_vert[dtid.x] = ~0U;
    }
}
