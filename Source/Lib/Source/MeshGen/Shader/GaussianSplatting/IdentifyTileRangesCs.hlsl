// Copyright (c) 2026 Minmin Gong
//

#include "Utils.hlslh"

static const uint32_t BlockDim = 256;

cbuffer param_cb
{
    uint32_t length;
};

Buffer<uint32_t2> point_keys_buff;

RWBuffer<uint32_t> ranges_buff;

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (dtid.x >= length)
    {
        return;
    }

    const uint32_t index = dtid.x;

    const uint32_t curr_tile = point_keys_buff[index].y;
    if (index == 0)
    {
        ranges_buff[curr_tile * 2 + 0] = 0;
    }
    else
    {
        const uint32_t prev_tile = point_keys_buff[index - 1].y;
        if (curr_tile != prev_tile)
        {
            ranges_buff[prev_tile * 2 + 1] = index;
            ranges_buff[curr_tile * 2 + 0] = index;
        }
    }
    if (index == length - 1)
    {
        ranges_buff[curr_tile * 2 + 1] = length;
    }
}
