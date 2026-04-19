// Copyright (c) 2026 Minmin Gong
//

#include "GSplatUtils.hlslh"
#include "Utils.hlslh"

static const uint32_t BlockDim = 256;

cbuffer param_cb
{
    uint32_t num_gaussians;
    uint32_t2 tile_grid;
};

Buffer<float2> screen_pos_buff;
Buffer<float> depth_buff;
Buffer<uint32_t> point_offset_buff;
Buffer<uint32_t> radius_buff;

RWBuffer<uint32_t2> point_keys_unsorted_buff;
RWBuffer<uint32_t> point_ids_unsorted_buff;

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID)
{
    [branch]
    if (dtid.x >= num_gaussians)
    {
        return;
    }

    const uint32_t index = dtid.x;
    if (radius_buff[index] > 0)
    {
        uint32_t offset = (index == 0) ? 0 : point_offset_buff[index - 1];
        uint2 rect_min, rect_max;
        GetRect(screen_pos_buff[index], radius_buff[index], rect_min, rect_max, tile_grid);

        // For each tile that the bounding rect overlaps, emit a key/value pair. The key is | tile ID |  depth  |,
        // and the value is the ID of the Gaussian.
        for (uint32_t y = rect_min.y; y < rect_max.y; ++y)
        {
            for (uint32_t x = rect_min.x; x < rect_max.x; ++x)
            {
                uint32_t2 key;
                key.y = y * tile_grid.x + x;
                key.x = asuint(depth_buff[index]);
                point_keys_unsorted_buff[offset] = key;
                point_ids_unsorted_buff[offset] = index;
                ++offset;
            }
        }
    }
}
