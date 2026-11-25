// Copyright (c) 2026 Minmin Gong
//

static const uint32_t BlockDim = 256;

#include "SumBlock.hlslh"

cbuffer param_cb
{
    float4 viewport;
    uint32_t num_gaussians;
};

Buffer<float4> screen_pos_extents_buff;

RWBuffer<uint32_t> pixels_counter_buff;

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID, uint32_t group_index : SV_GroupIndex)
{
    uint32_t pixels_touched = 0;

    [branch]
    if (dtid.x < num_gaussians)
    {
        const uint32_t index = dtid.x;
        const float4 screen_pos_extents = screen_pos_extents_buff[index];
        if (any(screen_pos_extents_buff[index] > 0))
        {
            const float2 screen_pos = screen_pos_extents.xy + 0.5f;
            const float2 adaptive_radius = screen_pos_extents.zw;

            const float2 beg_pos = clamp(screen_pos - adaptive_radius, 0, viewport.zw);
            const float2 end_pos = clamp(screen_pos + adaptive_radius, 0, viewport.zw);
            pixels_touched = ceil((end_pos.x - beg_pos.x) * (end_pos.y - beg_pos.y));
        }
    }

    const uint32_t sum = SumBlock(group_index, pixels_touched);
    if (group_index == 0)
    {
        InterlockedAdd(pixels_counter_buff[0], sum);
    }
}
