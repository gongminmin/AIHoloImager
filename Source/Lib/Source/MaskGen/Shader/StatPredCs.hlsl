// Copyright (c) 2024 Minmin Gong
//

static const uint32_t BlockDim = 16;
static const uint32_t MinWaveSize = 32;

cbuffer param_cb : register(b0)
{
    uint32_t2 texture_size;
};

Texture2D<float> input_tex : register(t0);

RWTexture2D<uint32_t> min_max_tex : register(u0);

groupshared float2 group_wave_min_max[BlockDim * BlockDim / MinWaveSize];

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID, uint32_t group_index : SV_GroupIndex)
{
    uint32_t2 coord = clamp(uint32_t2(dtid.xy), 0, texture_size - 1);
    float data = input_tex.Load(uint32_t3(coord, 0));
    GroupMemoryBarrierWithGroupSync();

    uint32_t wave_size = WaveGetLaneCount();
    uint32_t wave_index = group_index / wave_size;
    uint32_t lane_index = WaveGetLaneIndex();

    float2 min_max = float2(WaveActiveMin(data), WaveActiveMax(data));
    if (WaveIsFirstLane())
    {
        group_wave_min_max[wave_index] = min_max;
    }
    GroupMemoryBarrierWithGroupSync();

    uint32_t group_mem_size = BlockDim * BlockDim / wave_size;

    if (group_index < group_mem_size)
    {
        min_max = group_wave_min_max[group_index];
    }
    GroupMemoryBarrier();

    if (group_index < group_mem_size)
    {
        min_max = float2(WaveActiveMin(min_max.x), WaveActiveMax(min_max.y));
    }

    if (group_index == 0)
    {
        InterlockedMin(min_max_tex[uint32_t2(0, 0)], uint32_t(min_max.x * 1e5f));
        InterlockedMax(min_max_tex[uint32_t2(1, 0)], uint32_t(min_max.y * 1e5f));
    }
}
