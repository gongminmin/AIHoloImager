// Copyright (c) 2026 Minmin Gong
//

static const uint32_t BlockDim = 256;

cbuffer param_cb
{
    float4 viewport;
    uint32_t num_gaussians;
};

Buffer<float4> screen_pos_extents_buff;

RWBuffer<uint32_t> pixels_counter_buff;

static const uint32_t MinWaveSize = 16;
groupshared uint32_t group_sum[BlockDim / MinWaveSize];

uint32_t SumBlock(uint32_t group_index, uint32_t value)
{
    const uint32_t lane_index = WaveGetLaneIndex();
    const uint32_t wave_size = WaveGetLaneCount();
    const uint32_t wave_index = group_index / wave_size;
    const uint32_t num_waves = BlockDim / wave_size;

    const uint32_t sum = WaveActiveSum(value);
    if (lane_index == 0)
    {
        group_sum[wave_index] = sum;
    }
    GroupMemoryBarrierWithGroupSync();

    if (group_index < num_waves)
    {
        const uint32_t wave_sum = group_sum[group_index];
        group_sum[0] = WaveActiveSum(wave_sum);
    }
    GroupMemoryBarrierWithGroupSync();

    return group_sum[0];
}

[numthreads(BlockDim, 1, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID, uint32_t group_index : SV_GroupIndex)
{
    uint32_t pixels_touched = 0;

    [branch]
    if (dtid.x < num_gaussians)
    {
        const uint32_t index = dtid.x;
        const float4 screen_pos_extents = screen_pos_extents_buff[index];

        const float2 screen_pos = screen_pos_extents.xy + 0.5f;
        const float2 adaptive_radius = screen_pos_extents.zw;

        const float2 beg_pos = clamp(screen_pos - adaptive_radius, 0, viewport.zw);
        const float2 end_pos = clamp(screen_pos + adaptive_radius, 0, viewport.zw);
        pixels_touched = ceil((end_pos.x - beg_pos.x) * (end_pos.y - beg_pos.y));
    }

    const uint32_t sum = SumBlock(group_index, pixels_touched);
    if (group_index == 0)
    {
        InterlockedAdd(pixels_counter_buff[0], sum);
    }
}
