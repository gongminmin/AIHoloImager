// Copyright (c) 2024-2025 Minmin Gong
//

#include "Utils.hlslh"

static const uint32_t BlockDim = 16;
static const uint32_t MinWaveSize = 16;

cbuffer param_cb
{
    uint32_t2 texture_size;
};

Texture2D<float> input_tex;

RWBuffer<uint32_t> min_max_buff;

groupshared float2 group_wave_min_max[BlockDim * BlockDim / MinWaveSize];

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID, uint32_t group_index : SV_GroupIndex)
{
    uint32_t2 coord = clamp(uint32_t2(dtid.xy), 0, texture_size - 1);
    float data = input_tex.Load(uint32_t3(coord, 0));
    GroupMemoryBarrierWithGroupSync();

    const uint32_t wave_size = WaveGetLaneCount();
    const uint32_t wave_index = group_index / wave_size;

    float2 min_max = float2(WaveActiveMin(data), WaveActiveMax(data));

    uint32_t num_active_waves = DivUp(BlockDim * BlockDim, wave_size);
    while (num_active_waves > 1)
    {
        if (WaveIsFirstLane() && (wave_index < num_active_waves))
        {
            group_wave_min_max[wave_index] = min_max;
        }
        GroupMemoryBarrierWithGroupSync();

        if (group_index < num_active_waves)
        {
            min_max = group_wave_min_max[group_index];
        }
        GroupMemoryBarrier();

        if (group_index < num_active_waves)
        {
            min_max = float2(WaveActiveMin(min_max.x), WaveActiveMax(min_max.y));
        }

        num_active_waves = DivUp(num_active_waves, wave_size);
    }

    if (group_index == 0)
    {
        InterlockedMin(min_max_buff[0], uint32_t(min_max.x * 1e5f));
        InterlockedMax(min_max_buff[1], uint32_t(min_max.y * 1e5f));
    }
}
