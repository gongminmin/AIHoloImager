// Copyright (c) 2024 Minmin Gong
//

static const uint32_t BlockDim = 16;
static const uint32_t MinWaveSize = 32;

cbuffer param_cb : register(b0)
{
    uint32_t2 texture_size;
};

Texture2D<float4> input_tex : register(t0);

RWTexture2D<uint32_t> max_tex : register(u0);

groupshared float group_wave_max[BlockDim * BlockDim / MinWaveSize];

[numthreads(BlockDim, BlockDim, 1)]
void main(uint32_t3 dtid : SV_DispatchThreadID, uint32_t group_index : SV_GroupIndex)
{
    uint32_t2 coord = clamp(uint32_t2(dtid.xy), 0, texture_size - 1);
    float3 color = input_tex.Load(uint32_t3(coord, 0)).rgb;
    float data = max(max(color.r, color.g), color.b);
    GroupMemoryBarrierWithGroupSync();

    uint32_t wave_size = WaveGetLaneCount();
    uint32_t wave_index = group_index / wave_size;
    uint32_t lane_index = WaveGetLaneIndex();

    float max_ch = WaveActiveMax(data);
    if (WaveIsFirstLane())
    {
        group_wave_max[wave_index] = max_ch;
    }
    GroupMemoryBarrierWithGroupSync();

    uint32_t group_mem_size = BlockDim * BlockDim / wave_size;

    if (group_index < group_mem_size)
    {
        max_ch = group_wave_max[group_index];
    }
    GroupMemoryBarrier();

    if (group_index < group_mem_size)
    {
        max_ch = WaveActiveMax(max_ch);
    }

    if (group_index == 0)
    {
        InterlockedMax(max_tex[uint32_t2(0, 0)], uint32_t(max_ch * 1e5f));
    }
}
