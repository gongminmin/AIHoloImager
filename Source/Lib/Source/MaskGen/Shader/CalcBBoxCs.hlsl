// Copyright (c) 2024-2025 Minmin Gong
//

#include "Utils.hlslh"

static const uint32_t BlockDim = 16;
static const uint32_t MinWaveSize = 16;

cbuffer param_cb
{
    uint32_t2 texture_size;
};

Texture2D input_tex;

RWBuffer<uint32_t> bounding_box_buff;

groupshared uint32_t2 group_wave_mins[BlockDim * BlockDim / MinWaveSize];
groupshared uint32_t2 group_wave_maxs[BlockDim * BlockDim / MinWaveSize];

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID, uint32_t group_index : SV_GroupIndex)
{
    uint32_t2 bb_min = texture_size;
    uint32_t2 bb_max = uint32_t2(0, 0);
    if (all(dtid.xy < texture_size))
    {
        float alpha = input_tex.Load(uint32_t3(dtid.xy, 0)).a;
        if (alpha > 0.5f)
        {
            bb_min = dtid.xy;
            bb_max = dtid.xy;
        }
    }

    const uint32_t wave_size = WaveGetLaneCount();
    const uint32_t wave_index = group_index / wave_size;

    bb_min = WaveActiveMin(bb_min);
    bb_max = WaveActiveMax(bb_max);

    uint32_t num_active_waves = DivUp(BlockDim * BlockDim, wave_size);
    while (num_active_waves > 1)
    {
        if (WaveIsFirstLane() && (wave_index < num_active_waves))
        {
            group_wave_mins[wave_index] = bb_min;
            group_wave_maxs[wave_index] = bb_max;
        }
        GroupMemoryBarrierWithGroupSync();

        if (group_index < num_active_waves)
        {
            bb_min = group_wave_mins[group_index];
            bb_max = group_wave_maxs[group_index];
        }
        GroupMemoryBarrier();

        if (group_index < num_active_waves)
        {
            bb_min = WaveActiveMin(bb_min);
            bb_max = WaveActiveMax(bb_max);
        }

        num_active_waves = DivUp(num_active_waves, wave_size);
    }

    if ((group_index == 0) && (bb_min.x < bb_max.x) && (bb_min.y < bb_max.y))
    {
        InterlockedMin(bounding_box_buff[0], bb_min.x);
        InterlockedMin(bounding_box_buff[1], bb_min.y);
        InterlockedMax(bounding_box_buff[2], bb_max.x);
        InterlockedMax(bounding_box_buff[3], bb_max.y);
    }
}
